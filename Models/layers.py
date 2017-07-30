import tensorflow as tf
import numpy as np

def variable_on_cpu(name, shape, initializer, is_training=True, dtype=tf.float32):
  '''
  Create a shareable variable.
  '''

  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

# Thomas's hot new activation function
def selu(x, name):
  with tf.name_scope(name) as scope:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def cnn_layer(X, shape, strides, scope, padding='SAME', is_training=True):
  '''
  Create a convolution layer.
  '''

  kernel = variable_on_cpu(
    'kernel', shape, tf.contrib.layers.xavier_initializer(dtype=tf.float32), is_training=is_training
  )
  conv = tf.nn.conv2d(X, kernel, strides, padding=padding)
  biases = variable_on_cpu('b', [shape[-1]], tf.constant_initializer(0.0), is_training=is_training)
  activation = tf.nn.relu(conv + biases, name=scope.name)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation

def fc_layer(X, n_in, n_out, scope, is_training=True, act_func=None):
  '''
  Create a fully connected (multi-layer perceptron) layer.
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float64), is_training=is_training, dtype=tf.float64
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training, dtype=tf.float64)
  if act_func is not None:
    activation = act_func(tf.matmul(X, weights) + biases, name=scope.name)
  else:
    activation = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation

def fc_dropout_layer(X, n_in, n_out, scope, keep_prob, is_training=True, act_func=None):
  '''
  Create a fully connected (multi-layer perceptron) layer.
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float64), is_training=is_training, dtype=tf.float64
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training, dtype=tf.float64)
  if act_func is not None:
    activation = act_func(tf.matmul(X, weights) + biases, name=scope.name)
  else:
    activation = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  post_dropout = tf.nn.dropout(activation, keep_prob)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return post_dropout

def fc_bn_layer(X, n_in, n_out, scope, is_training=True, act_func=None):
  '''
  Create a fully connected layer with batch normalization
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float32), is_training=is_training
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0), is_training=is_training)
  fc = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=is_training, scope=scope)
  if act_func is not None:
    activation = act_func(fc, name=scope.name)
  else:
    activation = fc
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation


def lstm_layer(size):
    '''
    Create an LSTM layer
    :return:
    '''

    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

def lstm_dropout_layer(input, size, keep_prob, num_of_lstm_cells=1):
    '''
    Create an LSTM layer with dropout
    :return:
    '''
    all_lstm_cells = []
    for _ in range(num_of_lstm_cells):
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(size)
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
      all_lstm_cells.append(lstm_cell)

    if len(all_lstm_cells) > 1:
      final_lstm_cell = tf.contrib.rnn.MultiRNNCell(all_lstm_cells)
    else:
      final_lstm_cell = all_lstm_cells[0]

    # generate prediction
    outputs, _ = tf.contrib.rnn.static_rnn(final_lstm_cell, input, dtype=tf.float32)

    return outputs

def pool_layer(X, window_size, strides, scope, padding='SAME'):
  '''
  Create a max pooling layer.
  '''

  return tf.nn.max_pool(X, ksize=window_size, strides=strides,
                        padding=padding, name=scope.name)

def average_frames(input, stack_size):
  '''
  Average Frames
  '''
  Y = []
  nrows = input.shape[0] / stack_size
  for each in range(input.shape[1]):

    Y[each] = np.mean()