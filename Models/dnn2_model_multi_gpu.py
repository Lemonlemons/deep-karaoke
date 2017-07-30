from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re

class DNN2ModelMultiGpu(BaseModel):

  def build(self, stats):
    # dnn frame level
    self.spectograms_shape = (2050, )
    # dnn utterance level
    # self.fbanks_shape = (10, 48, 48)
    # dnn number of cells
    self.n_cells = 4100
    # batch size
    self.batch_size = 512 if self.is_training or self.is_testing else 1
    self.keep_prob = 0.5 if self.is_training else 1.0
    self.learning_rate = 1e-4

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      self.ORG_GUESSES, self.ORG_TRUTHS = self.read_gen_inputs([self.input_file], spectograms_shape=[np.prod(self.spectograms_shape),],
        batch_size=self.batch_size, capacity=100000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

      GUESS_MAX, TURTH_MAX = stats

      print('guess max: ' + str(GUESS_MAX))
      print('truth max: ' + str(TURTH_MAX))

      GUESSES = self.ORG_GUESSES
      TRUTHS = self.ORG_TRUTHS

      if not self.is_validating:
        # make them into "magnitude spectograms"
        GUESSES = tf.abs(GUESSES)
        TRUTHS = tf.abs(TRUTHS)

        # normalize them based on each frames individual maximum
        self.max_array = tf.reduce_max(TRUTHS, axis=1, keep_dims=True)
        self.NORM_TRUTHS = TRUTHS / self.max_array
        self.NORM_GUESSES = GUESSES / self.max_array
      else:
        self.NORM_TRUTHS = TRUTHS
        self.NORM_GUESSES = GUESSES

      # handle NAN's (basically when the maximum in a frame is zero
      nan_truth_array = tf.is_nan(self.NORM_TRUTHS)
      self.CORRECTED_TRUTHS = tf.where(nan_truth_array, x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)]), y=self.NORM_TRUTHS)
      nan_guess_array = tf.is_nan(self.NORM_GUESSES)
      self.CORRECTED_GUESSES = tf.where(nan_guess_array, x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)]), y=self.NORM_GUESSES)

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
                H_1 = fc_dropout_layer(self.CORRECTED_GUESSES, np.prod(self.spectograms_shape), self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_2') as innerscrope:
                H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_3') as innerscrope:
                H_3 = fc_dropout_layer(H_2, self.n_cells, self.n_cells, innerscrope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_4') as innerscrope:
                LOGITS = fc_layer(H_3, self.n_cells, np.prod(self.spectograms_shape), innerscrope, is_training=self.is_training)
              self.Y = tf.nn.sigmoid(LOGITS)

              # Compute the cross entropy loss.
              COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.CORRECTED_TRUTHS, logits=LOGITS
              ))

              tf.add_to_collection('losses', COST)

              # Assemble all of the losses for the current tower only.
              losses = tf.get_collection('losses', scope)

              # Calculate the total loss for the current tower.
              self.COST = tf.add_n(losses, name='total_loss')

              # Compute the accuracy
              self.ACCURACY = tf.subtract(1.0, tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(self.Y, self.CORRECTED_TRUTHS)), 10)))
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