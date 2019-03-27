import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from util.ops import shape_list
import ops



def prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i+1))
  return x

def reference_encoder(inputs, filters, kernel_size, strides, encoder_cell, is_training, scope='ref_encoder'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    ref_outputs = tf.expand_dims(inputs,axis=-1)
    # CNN stack
    for i, channel in enumerate(filters):
      ref_outputs = conv2d(ref_outputs, channel, kernel_size, strides, tf.nn.relu, is_training, 'conv2d_%d' % i)

    shapes = shape_list(ref_outputs)
    ref_outputs = tf.reshape(
      ref_outputs, 
      shapes[:-2] + [shapes[2] * shapes[3]])
    # RNN
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_cell,
      ref_outputs,
      dtype=tf.float32)

    reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 512, activation=tf.nn.tanh) # [N, 512]
    return reference_state


def encoder_cbhg(inputs, input_lengths, is_training):
  return cbhg(
    inputs,
    input_lengths,
    is_training,
    scope='encoder_cbhg',
    K=16,
    projections=[128, 128])


def post_cbhg(inputs, input_dim, is_training):
  return cbhg(
    inputs,
    None,
    is_training,
    scope='post_cbhg',
    K=8,
    projections=[256, input_dim])


def cbhg(inputs, input_lengths, is_training, scope, K, projections):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      # Convolution bank: concatenate on the last axis to stack channels from all convolutions
      conv_outputs = tf.concat(
        [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
        axis=-1
      )

    # Maxpooling:
    maxpool_output = tf.layers.max_pooling1d(
      conv_outputs,
      pool_size=2,
      strides=1,
      padding='same')

    # Two projection layers:
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

    # Residual connection:
    highway_input = proj2_output + inputs

    # Handle dimensionality mismatch:
    if highway_input.shape[2] != 128:
      highway_input = tf.layers.dense(highway_input, 128)

    # 4-layer HighwayNet:
    for i in range(4):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1))
    rnn_input = highway_input

    # Bidirectional RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(128),
      GRUCell(128),
      rnn_input,
      sequence_length=input_lengths,
      dtype=tf.float32)
    return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=128,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=128,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)

def conv2d(inputs, filters, kernel_size, strides, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv2d_output = tf.layers.conv2d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same')
    conv2d_output = tf.layers.batch_normalization(conv2d_output, training=is_training)
    if activation is not None:
      conv2d_output = activation(conv2d_output)
    return conv2d_output

def text_encoder(inputs, is_trianing):
    x = tf.nn.avg_pool(inputs, 2, 1)
    out = self.nn.dense(x, 512, activation = tf.tanh)
    return out

def image_encoder(inputs, is_training, norm, size):
    # CNN stack
    filters = [64, 128, 256, 512]
    x = inputs
    for i, channel in enumerate(filters):
      x = conv2d(rx, channel, 4, 2, tf.nn.relu, is_training, 'conv2d_%d' % i)

    outputs = tf.layers.max_pooling2d(x, [16, 16], padding='valid')    #[N, 512, 1]
   return outpus 

def modality_transformer(inputs, is_training):
    x = inputs
    x = tf.layers.dense(x, 256, activation=tf.nn.relu, name = 'dense_1')
    out = conv1d(x, 1, 256, tf.nn.relu, is_training = is_training)
    out = conn1d(out, 1, 256, tf.nn.relu, is_training = is_training)
    out += x
    return out
   
def modality_classifier(inputs, is_training):
    x = inputs
    x = tf.layers.dense(x, 128, activation = tf.nn.relu)
    logit = tf.layers.dense(x, 3)
    return logit

def image_decoder(input, is_training):
    batch_size = int(input.get_shape()[0])
    z = tf.layers.dense(input, 1024, activation=tf.nn.relu)
    G_input = tf.reshape(z, [batch_size, 4, 4, 64])     

    filters = [32, 16,8]
    
    for i, n in enumerate(filters):
        G = ops.deconv_block(G_input, n, 'CD{}_{}'.format(n, i), 4, 2, is_training, reuse = True, norm = 'batch', activation = 'relu')
        G = ops.deconv_block(G, 3, 'last_layer', 4, 2, is_training, reuse=True, norm=None, activation = 'tanh')

     return G

## text decoder scope

def decode(expanded_output):
    
    """ Decode the expanded output of the LSTM into a word. """
    config = self.config
    expanded_output = self.nn.dropout(expanded_output)
       
    # use 2 fc layers to decode
    temp = self.nn.dense(expanded_output,
                         units = config.dim_decode_layer,
                         activation = tf.tanh,
                         name = 'fc_1')
    temp = self.nn.dropout(temp)
    logits = self.nn.dense(temp,
                           units = config.vocabulary_size,
                           activation = None,
                           name = 'fc_2')
     return logits

def text_decoder(input, idx, is_training):
    with tf.variable_scope('D_txt') as scope: 
        # Setup the LSTM 
        lstm = tf.nn.rnn_cell.LSTMCell(
                128, initializer = tf.random_uniform_initializer(
                minval = -0.08, maxval = 0.08))
        lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm, 
                input_keep_prob = 1.0 - 0.3,
                outpt_keep_prob = 1.0 - 0.3,
                state_keep_prob = 1.0 - 0.3)

        # Initialize the LSTM using the mean context (input)
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(input, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if is_training:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = 20                #config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output

        # Generate the words one by one
        
        # Embed the last word        
        word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                 last_word)
        # Apply the LSTM
        current_input = tf.concat([context, word_embed], 1)
        output, state = lstm(current_input, last_state)
        memory, _ = state

         # Decode the expanded output of LSTM into a word
        expanded_output = tf.concat([output,
                                    context,
                                    word_embed],
                                    axis = 1)
        logits = decode(expanded_output)
        probs = tf.nn.softmax(logits)
        prediction = tf.argmax(logits, 1)
        predictions.append(prediction)
        
        # compute the loss for this step
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                #cross_entropies.append(masked_cross_entropy)
 
                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx] 

        return masked_cross_entropy
