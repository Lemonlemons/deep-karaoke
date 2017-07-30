from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re

class DNN1ModelMultiGpu(BaseModel):

  def build(self, stats):
    # dnn frame level
    # self.spectograms_shape = (1025, )
    self.spectograms_shape = (20, 1025)
    # dnn utterance level
    # self.fbanks_shape = (10, 48, 48)
    # dnn number of cells
    self.n_cells = np.prod(self.spectograms_shape)
    # batch size
    if self.is_testing or self.is_training:
      self.batch_size = 124
    elif self.is_generating:
      self.batch_size = 100
    else:
      self.batch_size = 1
    self.keep_prob = 1.0 if self.is_training else 1.0
    self.learning_rate = 1e-4

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      self.SPECTOGRAMS, self.MASKS = self.read_inputs([self.input_file],
        spectograms_shape=[np.prod(self.spectograms_shape),],
        batch_size=self.batch_size, capacity=10000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing or self.is_generating)
      )

      FIRST_SET_COUNT, SECOND_SET_COUNT = stats

      print('first set count: ' + str(FIRST_SET_COUNT))
      print('second set count: ' + str(SECOND_SET_COUNT))

      # make them into "magnitude spectograms"
      ABS_SPECTOGRAMS = tf.abs(self.SPECTOGRAMS)

      # normalize them based on each frames individual maximum
      self.max_array = tf.reduce_max(ABS_SPECTOGRAMS, axis=1, keep_dims=True)
      self.NORM_SPECTOGRAMS = ABS_SPECTOGRAMS / self.max_array

      # handle NAN's (basically when the maximum in a frame is zero) (can also be handled when creating the data)
      nan_array = tf.is_nan(self.NORM_SPECTOGRAMS)
      self.CORRECTED_SPECTOGRAMS = tf.where(nan_array, x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)], tf.float64), y=self.NORM_SPECTOGRAMS)

      # Compute gradients.
      OPTIMIZER = tf.train.AdamOptimizer(self.learning_rate)

      # Calculate the gradients for each model tower.
      tower_grads = []
      with tf.variable_scope(tf.get_variable_scope()):
        for i in range(self.num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('tower', i)) as scope:
              # Calculate the loss for one tower of the CIFAR model. This function
              # constructs the entire CIFAR model but shares the variables across
              # all towers.
              # Build feedforward layers.
              with tf.variable_scope('fully_connected_1') as innerscrope:
                H_1 = fc_dropout_layer(self.CORRECTED_SPECTOGRAMS, np.prod(self.spectograms_shape), self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_2') as innerscrope:
                H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_3') as innerscrope:
                H_3 = fc_dropout_layer(H_2, self.n_cells, self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_4') as innerscrope:
                LOGITS = fc_layer(H_3, self.n_cells, self.n_cells, innerscrope, is_training=self.is_training)
              self.Y = tf.nn.sigmoid(LOGITS)

              # Compute the cross entropy loss.
              COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.MASKS, logits=LOGITS
              ))

              tf.add_to_collection('losses', COST)

              # Assemble all of the losses for the current tower only.
              losses = tf.get_collection('losses', scope)

              # Calculate the total loss for the current tower.
              self.COST = tf.add_n(losses, name='total_loss')

              # Compute the accuracy
              self.ACCURACY = tf.subtract(tf.cast(1.0, tf.float64), tf.reduce_mean(tf.abs(tf.subtract(self.Y, self.MASKS))))
              tf.summary.scalar("accuracy", self.ACCURACY)

              for l in losses + [self.COST]:
                # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
                # session. This helps the clarity of presentation on tensorboard.
                loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
                tf.summary.scalar(loss_name, l)

              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()

              # Retain the summaries from the final tower.
              # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

              # Calculate the gradients for the batch of data on this CIFAR tower.
              grads = OPTIMIZER.compute_gradients(self.COST)

              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      GRADIENTS = self.average_gradients(tower_grads)

      # Apply gradients.
      self.APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(GRADIENTS)

      # Add histograms for gradients to our TensorBoard logs.
      for GRADIENT, VAR in GRADIENTS:
        if GRADIENT is not None:
          tf.summary.histogram('{}/gradients'.format(VAR.op.name), GRADIENT)

      # Collect the TensorBoard summaries.
      self.SUMMARIES_OP = tf.summary.merge_all()

  def average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads