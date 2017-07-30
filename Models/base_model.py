import csv

import tensorflow as tf
import time
import shutil
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants
import boto3
from preprocessing import *
import requests


class BaseModel(object):
  def __init__(self, args, stats, input_file, prod=False):
    self.mode = args.phase
    self.delete_old = args.delete_old == 'True'
    self.model = args.model
    self.num_gpus = int(args.num_gpus)
    self.model_file = 'Results/' + self.model + '/karaoke.ckpt'

    self.is_training = self.mode == 'train'
    self.is_testing = self.mode == 'test'
    self.is_generating = self.mode == 'gen'
    self.is_validating = self.mode == 'val'
    self.input_file = input_file
    self.num_batches = 9 # int(num_of_examples/self.batch_size)
    self.num_epochs = 10000

    self.input_pipeline_threads = 1
    # self.graph_config = tf.ConfigProto(allow_soft_placement=True,
    #                                    log_device_placement=False,
    #                                    inter_op_parallelism_threads=5,
    #                                    intra_op_parallelism_threads=2)
    self.graph_config = tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False)

    print('building Model')
    self.build(stats)

  def build(self, stats):
    raise NotImplementedError

  # training the chosen model
  def train(self):
    print('training Model')
    # Start training the model.
    # this session is for multi-gpu training
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      # Create Coordinator
      COORDINATOR = tf.train.Coordinator()

      # Initialize all the variables.
      SESSION.run(tf.global_variables_initializer())

      if self.delete_old:
        # remove old tensorboard and models files:
        shutil.rmtree('Results/'+self.model)
        os.makedirs('Results/'+self.model)
      else:
        # restore the session
        GRAPH_WRITER = tf.train.Saver()
        GRAPH_WRITER.restore(SESSION, self.model_file)

      shutil.rmtree('Tensorboard/' + self.model)
      os.makedirs('Tensorboard/' + self.model)

      # Start Queue Runners
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)
      # Create a tensorflow summary writer.
      SUMMARY_WRITER = tf.summary.FileWriter('Tensorboard/'+self.model, graph=self.GRAPH)
      # Create a tensorflow graph writer.
      GRAPH_SAVER = tf.train.Saver(tf.global_variables())

      TOTAL_DURATION = 0.0
      GLOBAL_STEP = 0
      BEST_ACC = 0.0
      for EPOCH in range(self.num_epochs):
        DURATION = 0
        ERROR = 0.0
        ACC = 0.0
        START_TIME = time.time()
        for MINI_BATCH in range(self.num_batches):
          _, SUMMARIES, COST_VAL, ACC = SESSION.run([
            self.APPLY_GRADIENT_OP, self.SUMMARIES_OP, self.COST, self.ACCURACY
          ])
          ERROR += COST_VAL
          GLOBAL_STEP += 1

        # Write the summaries to disk.
        SUMMARY_WRITER.add_summary(SUMMARIES, EPOCH)
        DURATION += time.time() - START_TIME
        TOTAL_DURATION += DURATION
        # Update the console.
        print('Epoch %d: loss = %.4f (%.3f sec) , acc = %.8f' % (EPOCH, ERROR, DURATION, ACC))
        if EPOCH % 25 == 0 and EPOCH != 0 and ACC > BEST_ACC:
          BEST_ACC = ACC
          print('Saving Session')
          GRAPH_SAVER.save(SESSION, self.model_file)
        if EPOCH == self.num_epochs or ERROR < 0.005:
          print(
            'Done training for %d epochs. (%.3f sec) total steps %d' % (EPOCH, TOTAL_DURATION, GLOBAL_STEP)
          )
          break
      if ACC > BEST_ACC:
        print('Saving Session')
        GRAPH_SAVER.save(SESSION, self.model_file)
      print('Training Done!')
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # generate the training data for dnn2
  def gen(self, gen_data_file, gen_meta_file, set_two_count):
    print('generating data')

    writer = tf.python_io.TFRecordWriter(gen_data_file)

    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      COORDINATOR = tf.train.Coordinator()
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      batch_count = set_two_count / self.batch_size
      print('set two count')
      print(set_two_count)
      print('batch count')
      print(batch_count)

      HIGHEST_GUESS = 0.0
      HIGHEST_TRUTH = 0.0

      for EPOCH in range(int(batch_count)):
        ESTIMATED_MASKS, SPECTOGRAMS, MASKS = SESSION.run([
          self.Y, self.SPECTOGRAMS, self.MASKS
        ])

        if (EPOCH % 500 == 0):
          print(EPOCH)

        for index, ESTIMATED_MASK in enumerate(ESTIMATED_MASKS):
          # create the masks for nonvocal
          estimated_nonvocal_mask = np.absolute(np.subtract(ESTIMATED_MASK, 1.0))

          # create the guessed vocal and nonvocal combo matrix
          guessed_vocals = np.multiply(ESTIMATED_MASK, SPECTOGRAMS[index])
          guessed_nonvocals = np.multiply(estimated_nonvocal_mask, SPECTOGRAMS[index])
          guessed_combo_matrix = np.concatenate((guessed_vocals, guessed_nonvocals))

          # multiply by 2 to return to the correct comparable size
          guessed_combo_matrix = np.multiply(guessed_combo_matrix, 2.0).flatten()

          # create the ground truth combo matrix
          # true_combo_matrix = np.concatenate((VOCALS, NONVOCALS), axis=1)

          # create nonvocal truth masks
          true_nonvocal_mask = np.absolute(np.subtract(MASKS[index], 1.0))

          # create the true vocal and nonvocal combo matrix
          true_vocals = np.multiply(MASKS[index], SPECTOGRAMS[index])
          true_nonvocals = np.multiply(true_nonvocal_mask, SPECTOGRAMS[index])
          true_combo_matrix = np.concatenate((true_vocals, true_nonvocals))

          # multiply by 2 to create true combo matrix size
          true_combo_matrix = np.multiply(true_combo_matrix, 2.0).flatten()

          largest_guess = np.amax(np.absolute(guessed_combo_matrix))
          largest_truth = np.amax(np.absolute(true_combo_matrix))

          if (largest_guess > HIGHEST_GUESS):
            HIGHEST_GUESS = largest_guess
          elif (largest_truth > HIGHEST_TRUTH):
            HIGHEST_TRUTH = largest_truth

          guessed_combo_matrix = guessed_combo_matrix.astype(np.float32, copy=False)
          true_combo_matrix = true_combo_matrix.astype(np.float32, copy=False)

          # Write the final input frames and binary_mask to disk.
          example = tf.train.Example(features=tf.train.Features(feature={
            'guess': bytes_feature(guessed_combo_matrix.flatten().tostring()),
            'truth': bytes_feature(true_combo_matrix.flatten().tostring())
          }))
          writer.write(example.SerializeToString())

      print('highest guess')
      print(HIGHEST_GUESS)
      print('highest truth')
      print(HIGHEST_TRUTH)

      with open(gen_meta_file, 'w') as OUTPUT:
        OUTPUT.write('{},{}'.format(HIGHEST_GUESS, HIGHEST_TRUTH))

      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

    writer.close()
    print('generation complete')

  def get_second_model_input(self, SPECTOGRAMS, configs):
    print('Getting input for Second Model')
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      ESTIMATED_MASKS = []
      COMBO_MATRICES = []
      VOCALS = []
      NONVOCALS = []
      PHASE_COMPONENTS = []

      print('size')
      print(len(SPECTOGRAMS))

      for SPECTOGRAM in SPECTOGRAMS:
        mask = SESSION.run(self.Y, feed_dict={self.spectograms: [SPECTOGRAM]})
        ESTIMATED_MASKS.append(mask)
        PHASE_COMPONENTS.append(SPECTOGRAM[-1].astype(np.float32))

      for index, ESTIMATED_MASK in enumerate(ESTIMATED_MASKS):
        # create the masks for nonvocal
        estimated_nonvocal_mask = np.absolute(np.subtract(ESTIMATED_MASK, 1.0))

        # create the guessed vocal and nonvocal combo matrix
        guessed_vocals = np.multiply(ESTIMATED_MASK, SPECTOGRAMS[index])
        guessed_nonvocals = np.multiply(estimated_nonvocal_mask, SPECTOGRAMS[index])
        guessed_combo_matrix = np.concatenate((guessed_vocals, guessed_nonvocals))

        # multiply by 2 to return to the correct comparable size
        guessed_combo_matrix = np.multiply(guessed_combo_matrix, 2.0).flatten()
        guessed_combo_matrix = guessed_combo_matrix.astype(np.float32, copy=False)
        COMBO_MATRICES.append(guessed_combo_matrix)

        vocal_matrix, nonvocal_matrix = np.split(guessed_combo_matrix, 2, axis=0)
        vocal_matrix = np.array([vocal_matrix], dtype=np.float32)
        nonvocal_matrix = np.array([nonvocal_matrix], dtype=np.float32)
        vocal_matrix[0][-1] = PHASE_COMPONENTS[index]
        nonvocal_matrix[0][-1] = PHASE_COMPONENTS[index]
        VOCALS.append(vocal_matrix)
        NONVOCALS.append(nonvocal_matrix)

      VOCAL_FRAMES = np.concatenate(VOCALS).T
      NONVOCAL_FRAMES = np.concatenate(NONVOCALS).T

      print('vocal and nonvocal array shapes')
      print(VOCAL_FRAMES.shape)
      print(NONVOCAL_FRAMES.shape)

      VOCAL_SIGNALS = istft(VOCAL_FRAMES, configs=configs)
      NONVOCAL_SIGNALS = istft(NONVOCAL_FRAMES, configs=configs)

      librosa.output.write_wav("vocal1.wav", VOCAL_SIGNALS, sr=configs['SAMPLE_RATE'])
      librosa.output.write_wav("nonvocal1.wav", NONVOCAL_SIGNALS, sr=configs['SAMPLE_RATE'])

      return COMBO_MATRICES, PHASE_COMPONENTS

  def refine_matrices_and_output(self, combo_matrices, phase_components, configs, stats):
    print('Getting input for Second Model')

    VOCAL_FRAMES = []
    NONVOCAL_FRAMES = []

    GUESS_MAX, TRUTH_MAX = stats

    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      print('size')
      print(len(combo_matrices))

      for index, combo_matrix in enumerate(combo_matrices):
        sign_matrix = np.sign(combo_matrix)
        norm_combo_matrix = np.absolute(combo_matrix)

        max_matrix = np.amax(norm_combo_matrix)
        corrected_combo_matrix = norm_combo_matrix / max_matrix

        refined_matrix, original = SESSION.run([self.Y, self.ORG_GUESSES], feed_dict={self.guesses: [corrected_combo_matrix]})
        refined_matrix = np.multiply(refined_matrix, max_matrix)
        refined_matrix = np.multiply(refined_matrix, sign_matrix)
        vocal_matrix, nonvocal_matrix = np.split(refined_matrix, 2, axis=1)
        vocal_matrix[0][-1] = phase_components[index]
        nonvocal_matrix[0][-1] = phase_components[index]
        VOCAL_FRAMES.append(vocal_matrix)
        NONVOCAL_FRAMES.append(nonvocal_matrix)
        if (index % 100 == 0):
          print(combo_matrix)
          print(refined_matrix)
          print(index)

    VOCAL_FRAMES = np.concatenate(VOCAL_FRAMES).T
    NONVOCAL_FRAMES = np.concatenate(NONVOCAL_FRAMES).T

    print('vocal and nonvocal array shapes')
    print(VOCAL_FRAMES.shape)
    print(NONVOCAL_FRAMES.shape)

    VOCAL_SIGNALS = istft(VOCAL_FRAMES, configs=configs)
    NONVOCAL_SIGNALS = istft(NONVOCAL_FRAMES, configs=configs)

    librosa.output.write_wav("vocal2.wav", VOCAL_SIGNALS, sr=configs['SAMPLE_RATE'])
    librosa.output.write_wav("nonvocal2.wav", NONVOCAL_SIGNALS, sr=configs['SAMPLE_RATE'])

  # get test accuracies of models
  def test(self):
    print('testing Model')
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      COORDINATOR = tf.train.Coordinator()
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      for EPOCH in range(10):
        ACC = SESSION.run(self.ACCURACY)
        # Update the console.
        print('Epoch %d: acc = %.8f' % (EPOCH, ACC))
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # for exporting the tensorflow model for tensorflow serving (not tested)
  def export(self):
    with tf.Session(graph=self.GRAPH) as SESSION:
      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      export_path = 'Results/SavedModels/'
      print('Exporting trained model to ', export_path)
      builder = saved_model_builder.SavedModelBuilder(export_path)

      # input_configs = {'x': tf.FixedLenFeature(shape=[np.prod(self.fbanks_shape),], dtype=tf.float32),}
      # tf_example = tf.parse_example(self.fbanks, input_configs)
      # x = tf.identity(tf_example['x'], name='x')

      inputs_fbanks = utils.build_tensor_info(self.fbanks)
      outputs_d_vector = utils.build_tensor_info(self.d_vector)

      prediction_signature = signature_def_utils.build_signature_def(
        inputs={'fbanks': inputs_fbanks},
        outputs={'d_vector': outputs_d_vector},
        method_name=signature_constants.PREDICT_METHOD_NAME)

      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
        SESSION, [tag_constants.SERVING],
        signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
        },
        legacy_init_op=legacy_init_op)

      builder.save()

      print('Done Exporting!')

  # inputs stuff
  def read_inputs(self, file_paths, batch_size=64, capacity=1000, min_after_dequeue=900, num_threads=2,
                  spectograms_shape=None, is_training=True):

    with tf.name_scope('input'):
      # if training we use an input queue otherwise we use placeholders
      if is_training:
        # Create a file name queue.
        filename_queue = tf.train.string_input_producer(file_paths)
        reader = tf.TFRecordReader()
        # Read an example from the TFRecords file.
        _, example = reader.read(filename_queue)
        features = tf.parse_single_example(example, features={
          'spectograms': tf.FixedLenFeature([], tf.string),
          'masks': tf.FixedLenFeature([], tf.string)
        })
        # Decode sample
        spectogram = tf.decode_raw(features['spectograms'], tf.float64)
        spectogram.set_shape(spectograms_shape)
        mask = tf.decode_raw(features['masks'], tf.float64)
        mask.set_shape(spectograms_shape)

        self.spectograms, self.masks = tf.train.shuffle_batch(
          [spectogram, mask], batch_size=batch_size,
          capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=num_threads,
        )
      else:
        spectograms_shape = [None] + spectograms_shape
        self.spectograms = tf.placeholder(tf.float64, shape=spectograms_shape)
        self.masks = tf.placeholder(tf.float64, shape=spectograms_shape)

      return self.spectograms, self.masks

  def read_gen_inputs(self, file_paths, batch_size=64, capacity=1000, min_after_dequeue=900, num_threads=2,
                  spectograms_shape=None, is_training=True):

    with tf.name_scope('input'):
      # if training we use an input queue otherwise we use placeholders
      if is_training:
        # Create a file name queue.
        filename_queue = tf.train.string_input_producer(file_paths)
        reader = tf.TFRecordReader()
        # Read an example from the TFRecords file.
        _, example = reader.read(filename_queue)
        features = tf.parse_single_example(example, features={
          'guess': tf.FixedLenFeature([], tf.string),
          'truth': tf.FixedLenFeature([], tf.string)
        })
        # Decode sample
        guesses = tf.decode_raw(features['guess'], tf.float32)
        guesses.set_shape(spectograms_shape)
        truths = tf.decode_raw(features['truth'], tf.float32)
        truths.set_shape(spectograms_shape)

        self.guesses, self.truths = tf.train.shuffle_batch(
          [guesses, truths], batch_size=batch_size, num_threads=num_threads,
          capacity=capacity, min_after_dequeue=min_after_dequeue
        )
      else:
        spectograms_shape = [None] + spectograms_shape
        self.guesses = tf.placeholder(tf.float32, shape=spectograms_shape)
        self.truths = tf.placeholder(tf.float32, shape=spectograms_shape)

      return self.guesses, self.truths
