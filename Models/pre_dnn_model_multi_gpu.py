from .base_model import *
from .layers import *
import os
import numpy as np
import tensorflow as tf
import math
import re

class DNNModelMultiGpu(BaseModel):

  def build(self):
    # dnn frame level
    self.windows_shape = (20, 1025)
    # dnn utterance level
    # self.fbanks_shape = (10, 48, 48)
    # dnn number of cells
    self.n_cells = np.prod(self.windows_shape)
    # batch size
    self.batch_size = 10 if self.is_training or self.is_testing else 1

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      WINDOWS, self.MASKS = self.read_inputs([self.input_file], windows_shape=[np.prod(self.windows_shape),],
        batch_size=self.batch_size, capacity=1000000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

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
              with tf.variable_scope('fully_connected_1') as innerscope:
                H_1 = fc_dropout_layer(WINDOWS, np.prod(self.windows_shape), self.n_cells, innerscope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_2') as innerscope:
                H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, innerscope, self.keep_prob, is_training=self.is_training, act_func=selu)
              with tf.variable_scope('fully_connected_3') as innerscope:
                LOGITS = fc_layer(H_2, self.n_cells, self.n_cells, innerscope, is_training=self.is_training)
              self.Y = tf.nn.sigmoid(LOGITS)

              # Compute the cross entropy loss.
              COST = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=LOGITS, labels=self.MASKS
              ))

              tf.add_to_collection('losses', COST)

              # Assemble all of the losses for the current tower only.
              losses = tf.get_collection('losses', scope)

              # Calculate the total loss for the current tower.
              self.COST = tf.add_n(losses, name='total_loss')

              # Compute the accuracy
              y_rounded = tf.round(self.Y)
              mask_rounded = tf.round(self.MASKS)
              binary_mask = tf.equal(y_rounded, mask_rounded)
              int_mask = tf.cast(binary_mask, tf.float32)
              sum = tf.reduce_sum(int_mask, axis=1)
              print(sum.shape)
              # Compute the accuracy
              self.ACCURACY = tf.divide(tf.reduce_mean(sum), self.n_cells)
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