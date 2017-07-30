from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re
from scipy.spatial import distance

class DNNModel(BaseModel):

  def build(self):
    # dnn frame level
    # self.windows_shape = (20, 1025)
    # dnn utterance level
    # self.fbanks_shape = (451, 21, 48)
    # dnn number of cells
    self.n_cells = np.prod(self.windows_shape)
    # batch size
    self.batch_size = 2048 if self.is_training or self.is_testing else 1

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      WINDOWS, self.MASKS = self.read_inputs([self.input_file], windows_shape=[np.prod(self.windows_shape),],
        batch_size=self.batch_size, capacity=100000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

      # Build feedforward layers.
      # This first layer is supposed to be a "locally_connected" layer however tensorflow doesn't have an implementation of that.
      with tf.variable_scope('fully_connected_1') as scope:
        H_1 = fc_dropout_layer(WINDOWS, np.prod(self.windows_shape), self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_2') as scope:
        H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_3') as scope:
        self.LOGITS = fc_layer(H_2, self.n_cells, self.n_cells, scope, is_training=self.is_training)
      self.Y = tf.nn.sigmoid(self.LOGITS)

      print(self.LOGITS.shape)

      # Compute the cross entropy loss.
      self.COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.MASKS, logits=self.LOGITS
      ))

      tf.summary.scalar("cost", self.COST)

      # one_hot_labels = tf.one_hot(self.LABELS, self.n_classes)

      y_rounded = tf.round(self.Y)
      mask_rounded = tf.round(self.MASKS)
      binary_mask = tf.equal(y_rounded, mask_rounded)
      int_mask = tf.cast(binary_mask, tf.float32)
      sum = tf.reduce_sum(int_mask, axis=1)
      print(sum.shape)
      # Compute the accuracy
      self.ACCURACY = tf.divide(tf.reduce_mean(sum), self.n_cells)
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