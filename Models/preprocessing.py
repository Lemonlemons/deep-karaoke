import random
import requests

import os

from audioread import NoBackendError
import audioread

import numpy as np
import tensorflow as tf

import csv
import sys
import json

import librosa

from scipy import signal


def prepare_train_and_test_data(configs, args):
  train_one_path = configs['TF_RECORDS_TRAIN_1']
  train_two_path = configs['TF_RECORDS_TRAIN_2']
  meta_path = configs['TF_RECORDS_META']
  # If the data hasn't been preprocessed, then do it now.
  if not os.path.exists(train_one_path) \
      or not os.path.exists(train_two_path) \
      or not os.path.exists(meta_path):
    print('Preparing Training and Testing Data')

    # if not os.path.exists(train_one_path) or not os.path.exists(train_two_path):
    # Write the training set.

    # save only mix_max from dnn1 data and only combo from dnn2 data because they will use them respectively
    # first_count = prepare_karaoke_tfrecord(train_one_path, configs, 0, 75, 30)
    # second_count = prepare_karaoke_tfrecord(train_two_path, configs, 75, 150, 105, second_set=True)
    first_count = prepare_karaoke_tfrecord(train_one_path, configs, 0, 45, 0)
    second_count = prepare_karaoke_tfrecord(train_two_path, configs, 45, 90, 0)

    # if not os.path.exists(val_path):
    # Write the testing set.
    # prepare_karaoke_tfrecord(test_path, configs, 50, 63)

    # if not os.path.exists(val_path):
    # Write the validation set
    # prepare_karaoke_tfrecord(val_path, configs, 50, 63, is_val=True)

    with open(configs['TF_RECORDS_META'], 'w') as OUTPUT:
      OUTPUT.write('{},{}'.format(first_count, second_count))

    print('preprocessing completed')
    sys.exit()
  else:
    if args.model == 'dnn2' or args.model == 'dnn2_multi':
      stats = get_stats(configs['GEN_META_FILE'])
    else:
      stats = get_stats(configs['TF_RECORDS_META'])
    print('loaded training data from disk')
    return stats

def prepare_val_data(configs):

  samples, _ = librosa.load(configs['TEST_FILE'], sr=configs['SAMPLE_RATE'], mono=True)
  true_mix = np.array(samples, dtype=np.float32)
  true_mix_stft = stft(true_mix, configs).T

  stats = get_stats(configs['TF_RECORDS_META'])
  print('loaded validation data')
  return stats, true_mix_stft

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_stats(meta_file):
  with open(meta_file, 'r') as INPUT:
    META_DATA = INPUT.readline()
    tuple = [
      float(DATA_POINT) for DATA_POINT in META_DATA.split(',')
    ]
  return tuple

def prepare_karaoke_tfrecord(set_path, configs, set_start_int, set_stop_int, dataset_break, second_set=False):

  writer = tf.python_io.TFRecordWriter(set_path)

  VOCAL_FRAMES = []
  NONVOCAL_FRAMES = []
  MIX_FRAMES = []
  MASKS = []
  count = 0

  with open(configs['FIRST_DATASET_FILE']) as dataset_file:
    jsondata = json.load(dataset_file)

  dataset_base = configs['SECOND_DATASET']
  second_set_files = os.listdir(dataset_base)

  for track_index in range(set_start_int, set_stop_int):
    print('base index: ' + str(track_index))
    start_dataset_two = True

    # when using both datasets
    # start_dataset_two = False
    # if dataset_break <= track_index:
    #   track_index = track_index - dataset_break
    #   start_dataset_two = True

    # if second_set and not start_dataset_two:
    #   track_index -= 45
    # elif second_set and start_dataset_two:
    #   track_index += 45
    # print('true index: ' + str(track_index))

    vocal_stems = []
    nonvocal_stems = []

    if not start_dataset_two:
      mix = jsondata['mixes'][track_index]

      for vocal_stem in mix['target_stems']:
        stem = jsondata['base_path'] + vocal_stem
        samples, _ = librosa.load(stem, sr=configs['SAMPLE_RATE'], mono=True)
        vocal_stems.append(samples)
      for nonvocal_stem in mix['other_stems']:
        stem = jsondata['base_path'] + nonvocal_stem
        samples, _ = librosa.load(stem, sr=configs['SAMPLE_RATE'], mono=True)
        nonvocal_stems.append(samples)
    else:
      track_base_path = dataset_base + second_set_files[track_index]
      track_files = os.listdir(track_base_path)

      print(track_base_path)

      for stem in track_files:
        print(stem)
        stem_path = track_base_path + '/' + stem
        print(stem_path)
        samples, _ = librosa.load(stem_path, sr=configs['SAMPLE_RATE'], mono=True)
        if stem == 'vocals.wav':
          vocal_stems.append(samples)
        else:
          nonvocal_stems.append(samples)

    vocal_stems = np.array(vocal_stems)
    nonvocal_stems = np.array(nonvocal_stems)
    vocal_mix = mix_stems(vocal_stems)
    nonvocal_mix = mix_stems(nonvocal_stems)
    true_mix = np.add(np.multiply(vocal_mix, 0.5), np.multiply(nonvocal_mix, 0.5))

    vocal_stft = stft(vocal_mix, configs).T
    nonvocal_stft = stft(nonvocal_mix, configs).T
    true_mix_stft = stft(true_mix, configs).T

    # removing the complex portion of the data
    vocal_stft = np.array(vocal_stft, dtype=np.float64)
    nonvocal_stft = np.array(nonvocal_stft, dtype=np.float64)
    true_mix_stft = np.array(true_mix_stft, dtype=np.float64)

    vocal_mag = np.absolute(vocal_stft)
    nonvocal_mag = np.absolute(nonvocal_stft)

    # handles when both the vocal_mag and nonvocal_mag are zero
    with np.errstate(divide='ignore', invalid='ignore'):
      true_masks = np.true_divide(vocal_mag, np.add(vocal_mag, nonvocal_mag))
      true_masks[true_masks == np.inf] = 0
      true_masks = np.nan_to_num(true_masks, copy=False)

    print('before')
    print(true_mix_stft.shape)
    print(true_masks.shape)

    # removing any portions of the vocal, nonvocal, or mix samples that just straight up zeros
    zeros_array = np.zeros(true_mix_stft[0].shape)
    mix_indexs_to_remove = []
    for index, frame in enumerate(true_mix_stft):
      if np.array_equal(zeros_array, frame):
        mix_indexs_to_remove.append(index)
    true_mix_stft = np.delete(true_mix_stft, mix_indexs_to_remove, 0)
    true_masks = np.delete(true_masks, mix_indexs_to_remove, 0)

    mask_indexs_to_remove = []
    for index, frame in enumerate(true_masks):
      if np.array_equal(zeros_array, frame):
        mask_indexs_to_remove.append(index)
    true_mix_stft = np.delete(true_mix_stft, mask_indexs_to_remove, 0)
    true_masks = np.delete(true_masks, mask_indexs_to_remove, 0)

    # pre-update
    hop_sample = configs['SAMPLE_HOP']

    mask_frames = sample_frames(true_masks, configs['STACKED_FRAMES'], hop_sample)
    mix_frames = sample_frames(true_mix_stft, configs['STACKED_FRAMES'], hop_sample)

    mask_frames = np.reshape(mask_frames, (len(mask_frames), 20500))
    mix_frames = np.reshape(mix_frames, (len(mix_frames), 20500))

    # important to notice that we're saving the non-absolute values
    # VOCAL_FRAMES.append(vocal_stft)
    # NONVOCAL_FRAMES.append(nonvocal_stft)
    # MIX_FRAMES.append(true_mix_stft)
    # MASKS.append(true_masks)
    print('after')
    print(mix_frames.shape)
    print(mask_frames.shape)

    count += len(true_mix_stft)

    for innerindex, window in enumerate(true_mix_stft):
      # Write the final input frames and binary_mask to disk.
      example = tf.train.Example(features=tf.train.Features(feature={
        'spectograms': bytes_feature(window.flatten().tostring()),
        'masks': bytes_feature(true_masks[innerindex].flatten().tostring())
      }))
      writer.write(example.SerializeToString())

  # need this if I want the max of all frames
  # MIX_FRAMES = np.concatenate(MIX_FRAMES)
  # MASKS = np.concatenate(MASKS)
  #
  # count, _ = MIX_FRAMES.shape
  #
  # print("finalizing processing")
  # # ideal_binary_mask = np.greater(VOCAL_FRAMES, NONVOCAL_FRAMES)
  # ideal_binary_mask = MASKS
  #
  # training_input = MIX_FRAMES.astype(np.float64, copy=False)
  # ideal_binary_mask = ideal_binary_mask.astype(np.float64, copy=False)
  #
  # for innerindex, window in enumerate(training_input):
  #   # Write the final input frames and binary_mask to disk.
  #   example = tf.train.Example(features=tf.train.Features(feature={
  #     'spectograms': bytes_feature(window.flatten().tostring()),
  #     'masks': bytes_feature(ideal_binary_mask[innerindex].flatten().tostring())
  #   }))
  #   writer.write(example.SerializeToString())

  writer.close()
  return count

def stft(mix, configs):
  fft_size = configs['FFT_SIZE']
  amount_of_overlap = fft_size - configs['HOP_SIZE']
  _, _, Zxx = signal.stft(mix, window='hann', nperseg=fft_size, noverlap=amount_of_overlap, nfft=fft_size)
  return Zxx

def istft(mix, configs):
  fft_size = configs['FFT_SIZE']
  amount_of_overlap = fft_size - configs['HOP_SIZE']
  _, x = signal.istft(mix, window='hann', nperseg=fft_size, noverlap=amount_of_overlap, nfft=fft_size)
  return x

def mix_stems(stems):
  nstems, nsamples = stems.shape
  mix = np.zeros((nsamples, ))
  for stem in stems:
    stem = np.divide(stem, np.amax(np.absolute(stem)))
    stem = np.divide(stem, nstems)
    mix = np.add(mix, stem)
  return mix

def mix_stems_max(stems):
  nstems, nsamples = stems.shape
  mix = np.zeros((nsamples, ))
  big_max = np.amax(np.absolute(stems))
  for stem in stems:
    stem = np.divide(stem, big_max)
    stem = np.divide(stem, nstems)
    mix = np.add(mix, stem)
  return mix

def sample_frames(X, L, H):
  n_hops = np.round((X.shape[0] - L) / H)
  Y = []
  for hop in range(n_hops):
    hop_start = (hop * H)
    chunk = X[hop_start:hop_start + L, :]
    Y.append(chunk)
  return np.array(Y, dtype=np.float64)

