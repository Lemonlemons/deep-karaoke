from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re
from scipy.spatial import distance

class DNN1Model(BaseModel):

  def build(self, stats):
    # dnn frame level
    # self.spectograms_shape = (1025, )
    self.spectograms_shape = (20, 1025)
    # dnn utterance level
    # self.fbanks_shape = (451, 21, 48)
    # dnn number of cells
    self.n_cells = np.prod(self.spectograms_shape)
    # batch size
    if self.is_testing or self.is_training:
      self.batch_size = 1
    elif self.is_generating:
      self.batch_size = 100
    else:
      self.batch_size = 1
    self.keep_prob = 1.0 if self.is_training else 1.0
    self.learning_rate = 1e-4

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      self.SPECTOGRAMS, self.MASKS = self.read_inputs([self.input_file], spectograms_shape=[np.prod(self.spectograms_shape),],
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
      self.CORRECTED_SPECTOGRAMS = tf.where(nan_array,
                                            x=tf.zeros([self.batch_size, np.prod(self.spectograms_shape)], tf.float64),
                                            y=self.NORM_SPECTOGRAMS)

      # Build feedforward layers.
      # This first layer is supposed to be a "locally_connected" layer however tensorflow doesn't have an implementation of that.
      with tf.variable_scope('fully_connected_1') as scope:
        H_1 = fc_dropout_layer(self.CORRECTED_SPECTOGRAMS, np.prod(self.spectograms_shape), self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_2') as scope:
        H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_3') as scope:
        H_3 = fc_dropout_layer(H_2, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_4') as scope:
        self.LOGITS = fc_layer(H_3, self.n_cells, self.n_cells, scope, is_training=self.is_training)
      self.Y = tf.nn.sigmoid(self.LOGITS)

      print(self.LOGITS.shape)

      # Compute the cross entropy loss.
      self.COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.MASKS, logits=self.LOGITS
      ))

      tf.summary.scalar("cost", self.COST)

      # Compute the accuracy
      self.ACCURACY = tf.subtract(tf.cast(1.0, tf.float64), tf.reduce_mean(tf.abs(tf.subtract(self.Y, self.MASKS))))
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