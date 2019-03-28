import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet, reference_encoder, text_encoder, image encoder\
modality_trainsformer, modality_classifier, image_decoder, decode, text_decoder
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, ZoneoutWrapper
from .multihead_attention import MultiheadAttention
#from .img_encoder import image_encoder


class SkipNet():
    
    def __init__(self, hparams):
        self._hparams = hparams


    def initialize(self, txt_targets_A, txt_lenth_A, txt_targets_B, txt_lenth_B, mel_targets, image_targets):
        #with tf.variable_scope('inference') as scope:
        is_training = mel_targets is not None
        #is_teacher_force_generating = mel_targets is not None
        batch_size = tf.shape(mel_targets)[0]
        hp = self._hparams
        
        # Embeddings for text
        embedding_table = tf.get_variable(
          'text_embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=0.5))
        embedded_txt_inputs_A = tf.nn.embedding_lookup(embedding_table, txt_targets_A)            #[N, T_in, 128]
        embedded_txt_inputs_B = tf.nn.embedding_lookup(embedding_tabel, txt_targets_B)
        
        
    #------------------------ Encoder Scope----------------------------------------------
        # 'e space': outputs from Modality Encoders

        # Text Encoder
        with tf.variable_scope('E_text', reuse = tf.AUTO_REUSE) as scope: 
            prenet_outputs_A = prenet(embedded_txt_inputs_A, is_training)                       # [N, T_in, 128]
            prenet_outputs_B = prenet(embedded_txt_inputs_B, is_training)                       # [N, T_in, 128]
        
            cbhg_outputs_A = encoder_cbhg(prenet_outputs_A, input_lengths_A, is_training)  
            cbhg_outputs_B = encoder_cbhg(prenet_outputs_B, input_lengths_B, is_training)  
            
            txt_encoder_outputs_A = text_encoder(cbhg_outputs_A, is_training)
            txt_encoder_outputs_B = text_encoder(cbhg_outputs_B, is_training)

            self.e_txt_A = txt_encoder_outputs_A
            self.e_txt_B = txt_encoder_outputs_B

        # Speech Encoder 
        with tf.variable_scope('E_speech', reuse = tf.AUTO_REUSE) as scope:
            speech_outputs = reference_encoder(
                mel_targets, 
                filters=hp.reference_filters, 
                kernel_size=(3,3),
                strides=(2,2),
                encoder_cell=GRUCell(hp.reference_depth),
                is_training=is_training)                                                 # [N, 256]
            self.e_speech = speech_outputs                                       

        # Image Encoder
        with tf.variable_scope('E_image', reuse = tf.AUTO_REUSE) as scope:
            img_outputs = image_encoder( 
                is_training=is_training,
                norm='batch',
                image_size = 128)
            self.e_img = img_outputs  
    
     #-------------------------Universal Computing Body------------------------------------

        # Modality Transformer T
        with tf.variable_scope('T', reuse = tf.AUTO_REUSE) as scope:
            # 'z space': output from Modality Transformer
            self.z_img = modality_transformer(self.e_img, is_training = is_training)
            self.z_txt_A = modality_transformer(self.e_txt_A, is_training = is_training)
            self.z_txt_B = modality_transformer(self.e_txt_B, is_training = is_training)
            self.z_speech = modality_transformer(self.e_speech, is_training = is_training)

        # Modality Classifier C
        with tf.variable_scope('C', reuse = tf.AUTO_REUSE) as scope:
            self.c_logit_img = modality_classifier(self.z_img, is_training = is_training)
            c_logit_txt_A = modality_classifier(self.z_txt_A, is_training = is_training)
            c_logit_txt_B = modality_classifier(self.z_txt_B, is_training = is_training)
            self.c_logit_txt = c_logit_txt_A + c_logit_txt_B
            self.c_logit_speech = modality_classifier(self.z_speech, is_training =is_training)
             
        # Memory Fusion Module M
        with tf.variable_scope('M', reuse = tf.AUDO_REUSE) as scope:
            # Global tokens
            tokens = tf.get_variable(
            'global_tokens', [hp.num_gst, hp.style_embed_depth // hp.num_heads], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.5))
            self.tokens = tokens
                
          # Multi-head Attention
            attention_img = MultiheadAttention(
            tf.expand_dims(self.z_img,axis=1),                                   # [N, 1, 256]
            tf.tanh(tf.tile(tf.expand_dims(tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=hp.style_att_dim,
            attention_type=hp.style_att_type)

            attention_speech = MultiheadAttention(
            tf.expand_dims(self.z_speech,axis=1),                                   # [N, 1, 256]
            tf.tanh(tf.tile(tf.expand_dims(tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=hp.style_att_dim,
            attention_type=hp.style_att_type)

            attention_txt_A = MultiheadAttention(
            tf.expand_dims(self.z_txt_A, axis=1),                                
            tf.tanh(tf.tile(tf.expand_dims(tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=hp.style_att_dim,
            attention_type=hp.style_att_type)

            attention_txt_B = MultiheadAttention(
            tf.expand_dims(self.z_txt_B, axis=1),                   
            tf.tanh(tf.tile(tf.expand_dims(tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
            num_heads=hp.num_heads,
            num_units=hp.style_att_dim,
            attention_type=hp.style_att_type)

            output_img = attention_img.multi_head_attention()                   # [N, 1, 256]
            output_txt_A = attention_txt_A.multi_head_attention()
            output_txt_B = attention_txt_B.multi_head_attention()
            output_speech = attention_speech.multi_head_attention()

            # 'u space': output form Memory Fusion Module
            self.u_img = output_img
            self.u_speech = output_speech
            self.u_txt_A = output_txt_A
            self.u_txt_B = output_txt_B   

        
        #---------------Decoder Scopt---------------------------------------------------------            
           
        # Image Decoder scope
        with tf.variable_scope('D_img') as scope:
            fake_img = image_decoder( self.u_img, is_train = self.is_training)
            self.fake_img = fake_img

   
        # Speech Decoder scope
        with tf.variable_scope('D_speech') as scope: 
            # Attention
            attention_cell = AttentionWrapper(
              GRUCell(hp.attention_depth),
              BahdanauAttention(hp.attention_depth, self.u_speech, memory_sequence_length=input_lengths),
              alignment_history=True,
              output_attention=False)                                                  # [N, T_in, 256]

            # Concatenate attention context vector and RNN cell output.
            concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              

            # Decoder (layers specified bottom to top):
            decoder_cell = MultiRNNCell([
                OutputProjectionWrapper(concat_cell, hp.rnn_depth),
                ResidualWrapper(ZoneoutWrapper(LSTMCell(hp.rnn_depth), 0.1)),
                ResidualWrapper(ZoneoutWrapper(LSTMCell(hp.rnn_depth), 0.1))
              ], state_is_tuple=True)                                                  # [N, T_in, 256]

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)
            decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            if is_training:
              helper = TacoTrainingHelper(inputs, mel_targets, hp)
            
            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
              BasicDecoder(output_cell, helper, decoder_init_state),
              maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]

            # Reshape outputs to be one output per entry
            fake_mel = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels]) # [N, T_out, M]
            self.fake_mel = fake_mel

          
        self.txt_targets_A = txt_targets_A
        self.txt_lengths_B = txt_lengths_B
        self.mel_targets = mel_targets
        self.image_targets = image_targets
        
    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams

            # image and speech regression with l1_loss 
            speech_loss_step = tf.reduce_mean(tf.abs(self.mel_targets - self.fake_mel))
            img_loss_step = tf.reduce_mean(tf.abs(self.image_targets -self.fake_image))

            cross_entropies_A = []
            cross_entropies_B = []
            # compute loss for each step 
            num_steps = self.txt_lenth_B
            for idx in range(num_stpes):    
                # text loss cross_entropy loss per-step
                cross_entropy_A = text_decoder(self.u_txt_A,idx, embedded_txt_inputs_A, is_train=self.is_training)
                cross_entropy_B = text_decoder(self.u_txt_B,idx, embedded_txt_inputs_B,
is_train=self.is_training)
                
            # compute final loss for text
            cross_entropies_A = tf.stack(cross_entropies_A, axis = 1)
            cross_entropy_loss_A = tf.reduce_sum(cross_entropies_A) 
            cross_entropies_B = tf.stack(cross_entropies_B, axis = 1)
            cross_entropy_loss_B = tf.reduce_sum(cross_entropies_B) 
            reg_loss = tf.losses.get_regularization_loss()
            txt_loss_A = cross_entropy_loss_A + reg_loss                     
            txt_loss_B = cross_entropy_loss_B + reg_loss
                                
            # domain classification loss
            txt_d_loss_A = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_txt_A, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
            txt_d_loss = tf.reduce_mean(txt_d_loss_A)
            txt_d_loss_B = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_txt_B, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
            txt_d_loss = tf.reduce_mean(txt_d_loss_B)

            img_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_img, labels=tf.constant([[0.0, 1.0, 0.0]] * batch_size))
            img_d_loss = tf.reduce_mean(img_d_loss)
            speech_d_loss =\
              tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_speech, labels=tf.constant([[0.0, 0.0, 1.0]] * batch_size))
            speech_d_loss = tf.reduce_mean(speech_d_loss)
            
            img_g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_img, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
            img_g_loss = tf.reduce_mean(img_g_loss)
            speech_g_loss =\
               tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_speech, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
            speech_g_loss = tf.reduce_mean(speech_g_loss)
            
            self.domain_d_loss = txt_d_loss_A + txt_d_loss_B (img_d_loss + speech_d_loss) * 2.
            self.domain_g_loss = (img_g_loss + speech_g_loss) * 2.

            self.recon_img_loss = img_loss_step
            self.recon_speech_loss = speech_loss_step
            self.recon_txt_loss_A = txt_loss_A
            self.recon_txt_loss_B = txt_loss_B
            self.recon_loss = (img_loss_step + speech_loss_step + txt_loss_A + txt_loss_B) * 10.
    
    def add_optimizer(self, global_step):
        '''Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
                        
            all_vars = tf.trainable_variables()
            gen_vars = [var for var in all_vars if not var.name.startswith('C_g') ]
            dis_vars = [var for var in all_vars if var.name.startswith('C_d')]      
            inf_vars = [var for var in all_vars if var.name.startswith('E') or\
                var.name.starswith('T') or var.name.starswith('M')]
            dec_vars = [var for var in all_vars if var.name.startswith('D_')]
                        
            
            # optimizer for reconstruction loss
            optimizer_recon = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients_recon, variables_recon = zip(*optimizer.compute_gradients(self.recon_loss + self.domain_g_loss ,
            var_list = gen_vars + inf_vars + dec_vars))
            self.gradients_recon = gradients_recon
            clipped_gradients_recon, _ = tf.clip_by_global_norm(gradients_recon, 1.0)
            
            # Add dependency on UPDATE_OPS:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
              self.optimize_recon = optimizer.apply_gradients(zip(clipped_gradients_recon, variables_recon),
                global_step=global_step)
            
            # optimizer for domain d loss
            optimizer_domain = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.domain_d_loss, var_list = d_vars))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            
            # Add dependency on UPDATE_OPS:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
              self.optimize_domain = optimizer.apply_gradients(zip(clipped_gradients, variables),
                global_step=global_step)

      


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
