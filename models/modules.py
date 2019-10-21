# Copyright (c) Microsoft Corporation. All rights reserved.

# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from util.ops import shape_list


def prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
  x = inputs
  # Dropout in both training and testing
  drop_rate = 0.5
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, training=True, name='dropout_%d' % (i+1))
  return x

def reference_encoder(inputs, filters, kernel_size, strides, encoder_cell, is_training, scope='ref_encoder'):
  with tf.variable_scope(scope):
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

    reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
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

def info_encoder(inputs, scope):
    with tf.vairable_scope(scope):
      A = tf.layers.dense(inputs,
                units=256,
                activation=tf.nn.relu,
                name = 'A')
      B = tf.layers.dense(inputs, 
                units = 256,
                activation = tf.nn.sigmoid,
                name = 'B',
                bias_initializer = tf.constant_initializer(-1.0))
      return A * B + inputs * (1.0 - B)
          

def domain_classifier(inputs, scope):
    with tf.variable_scope(scope):
      output = tf.layers.conv1d(
        inputs, 
        filters = 128,
        activation = tf.nn.relu)
      output = tf.layers.batch_normalization(output, training=is_training)
      logit = tf.layers.dense(outputs, units=3)
      return logit

    
