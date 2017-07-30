from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re
from scipy.spatial import distance

class DNN2Model(BaseModel):

  def build(self, stats):
    # dnn frame level
    self.spectograms_shape = (2050, )
    # dnn utterance level
    # self.fbanks_shape = (451, 21, 48)
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
        batch_size=self.batch_size, capacity=10000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

      GUESS_MAX, TURTH_MAX = stats

      print('guess max: ' + str(GUESS_MAX))
      print('truth max: ' + str(TURTH_MAX))

      GUESSES = self.ORG_GUESSES
      TRUTHS = self.ORG_TRUTHS

      # make them into "magnitude spectograms"
      GUESSES = tf.abs(GUESSES)
      TRUTHS = tf.abs(TRUTHS)

      # normalize them based on each frames individual maximum
      self.max_array = tf.reduce_max(TRUTHS, axis=1, keep_dims=True)
      self.NORM_TRUTHS = TRUTHS / self.max_array
      self.NORM_GUESSES = GUESSES / self.max_array

      # handle NAN's (basically when the maximum in a frame is zero
      nan_truth_array = tf.is_nan(self.NORM_TRUTHS)
      self.CORRECTED_TRUTHS = tf.where(nan_truth_array, x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)]), y=self.NORM_TRUTHS)
      nan_guess_array = tf.is_nan(self.NORM_GUESSES)
      self.CORRECTED_GUESSES = tf.where(nan_guess_array, x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)]), y=self.NORM_GUESSES)

      # Build feedforward layers.
      # This first layer is supposed to be a "locally_connected" layer however tensorflow doesn't have an implementation of that.
      with tf.variable_scope('fully_connected_1') as scope:
        H_1 = fc_dropout_layer(self.CORRECTED_GUESSES, np.prod(self.spectograms_shape), self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_2') as scope:
        H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_3') as scope:
        H_3 = fc_dropout_layer(H_2, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_4') as scope:
        self.LOGITS = fc_layer(H_3, self.n_cells, np.prod(self.spectograms_shape), scope, is_training=self.is_training)
      self.Y = tf.nn.sigmoid(self.LOGITS)

      print(self.LOGITS.shape)

      # Compute the cross entropy loss.
      self.COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.CORRECTED_TRUTHS, logits=self.LOGITS
      ))

      tf.summary.scalar("cost", self.COST)

      # Compute the accuracy
      self.ACCURACY = tf.subtract(1.0, tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(self.Y, self.CORRECTED_TRUTHS)), 10)))
      tf.summary.scalar("accuracy", self.ACCURACY)

      # Compute gradients.
      OPTIMIZER = tf.train.AdamOptimizer(self.learning_rate)

      GRADIENTS = OPTIMIZER.compute_gradients(self.COST)

      # Apply gradients.
      self.APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(GRADIENTS)

      # Add histograms for gradients to our TensorBoard logs.
      for GRADIENT, VAR in GRADIENTS:
        if GRADIENT is not None:
          tf.summary.histogram('{}/gradients'.format(VAR.op.name), GRADIENT)

      # Collect the TensorBoard summaries.
      self.SUMMARIES_OP = tf.summary.merge_all()