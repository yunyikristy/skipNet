# Copyright (c) Microsoft Corporation. All rights reserved.

# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet, reference_encoder
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, ZoneoutWrapper
from .multihead_attention import MultiheadAttention
from .img_encoder import image_encoder


class M3D():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, txt_targets, txt_lengths, mel_targets, image_targets):
    with tf.variable_scope('inference') as scope:
      is_training = mel_targets is not None
      is_teacher_force_generating = mel_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings for text
      embedding_table = tf.get_variable(
        'text_embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_txt_inputs = tf.nn.embedding_lookup(embedding_table, txt_targets)           # [N, T_in, 256]
      
       
      # Text Encoder
      prenet_outputs = prenet(embedded_txt_inputs, is_training)                       # [N, T_in, 128]
      txt_encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training)  # [N, T_in, 256]
      self.z_txt
     
      # Speech Encoder 
      speech_outputs = reference_encoder(
          mel_targets, 
          filters=hp.reference_filters, 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(hp.reference_depth),
          is_training=is_training)                                                 # [N, 256]
      self.z_speech = speech_outputs                                       

      # Image Encoder
      img_outputs = image_encoder('E', 
          is_training=is_training,
          norm='batch',
          image_size = 128)
      self.z_img = img_outputs  

            
      def global_body(self, input):
        # Global computing body (share weights)
        # information fusion encoder
        self.z_fuse = info_encoder(input)      # [N, 1, 256]
        # Global  tokens (GST)
        gst_tokens = tf.get_variable(
          'global_tokens', [hp.num_gst, hp.embed_depth // hp.num_heads], dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=0.5))
        self.gst_tokens = gst_tokens

        # Attention
        attention = MultiheadAttention(
          tf.expand_dims(z_fuse, axis=1),                                   # [N, 1, 256]
          tf.tanh(tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
          num_heads=hp.num_heads,
          num_units=hp.style_att_dim,
          attention_type=hp.style_att_type)

        output = attention.multi_head_attention()                   # [N, 1, 256]
        self.uni_embedding = output
        return self.uni_embedding

      # Domain classification network
      domain_logit_txt = domain_classifier('D',
          is_training = is_training,
          norm='batch',
          info_encoder(self.z_txt)) 
     
      domain_logit_img = domain_classifier('D',
          is_training = is_training,
          norm='batch',
          info_encoder(self.z_img)) 

      domain_logit_speech = domain_classifier('D',
          is_training = is_training,
          norm='batch',
          info_encoder(self.z_speech))  


    # out of inference scope
    # Add style embedding to every text encoder state

    # Text Decoder scope
    with tf.variable_scope('text_decoder') as scope:
      
      attention_cell = AttentionWrapper(
        GRUCell(hp.attention_depth),
        BahdanauAttention(hp.attention_depth, uni_embeddings, memory_sequence_length=input_lengths),
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

      output_cell = OutputProjectionWrapper(decoder_cell, hp.outputs_per_step)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      
      decoder_outputs, _  = tf.nn.dynamic_rnn(
        cell = output_cell, 
        initial_state = decoder_init_state,
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]
      with tf.variable_scope('text_logits') as scope:   
        txt_logit = tf.contrib.layers.fully_connected(
          inputs = decoder_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope = logits_scope)
        
      
    # Image Decoder scope
    with tf.variable_scope('image_decoder') as scope:
      G = Generator('G', is_train = self.is_training, norm = 'batch', 
                    image_size = 128)
      fake_img = G(uni_embeddings)

   
    # Speech Decoder scope
    with tf.variable_scope('speech_decoder') as scope: 
      # Attention
      attention_cell = AttentionWrapper(
        GRUCell(hp.attention_depth),
        BahdanauAttention(hp.attention_depth, uni_embeddings, memory_sequence_length=input_lengths),
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

      
    self.txt_targets = txt_targets
    self.txt_lengths = txt_lengths
    self.mel_targets = mel_targets
    self.image_targets = image_targets
    self.txt_targets = txt_targets
    self.txt_logit = txt_logit
    self.fake_mel = fake_mel
    self.fake_img = fake_img   

  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      recon_speech_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.fake_mel))
      recon_image_loss = tf.reduce_mean(tf.abs(self.image_targets -self.fake_image))
      recon_txt_loss = tf.reduce_sum(tf.nn.soft_max_cross_entropy_with_logits(logit=txt_logit, labels=txt_targets))
      self.recon_loss = recon_speech_loss + recon_image_loss + recon_txt_loss

      txt_d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=domain_logit_txt, labels=tf.constant([[1.0, 0.0, 0.0]] * batch_size))
      txt_d_loss = tf.reduce_mean(txt_d_loss)
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
      
      self.domain_d_loss = txt_d_loss + (img_d_loss + speech_d_loss) * 2.
      self.domain_g_loss = (img_g_loss + speech_g_loss) * 2.

      self.loss = self.recon_loss + self.domain_d_loss +self.domain_g_loss


  def add_optimizer(self, global_step):
    '''Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      
      all_vars = tf.trainable_variables()
      g_vars = [var for var in all_vars if not var.name.startswith('domain_') ]
      d_vars = [var for var in all_vars if var.name.startswith('domain_')]      

      # optimizer for reconstruction loss
      optimizer_recon = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.recon_loss + self.domain_g_loss ,
      var_list = g_vars))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS:
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize_recon = optimizer.apply_gradients(zip(clipped_gradients, variables),
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
