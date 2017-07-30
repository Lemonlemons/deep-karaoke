import ConfigParser
import argparse
import os

from Models.dnn1_model import *
from Models.dnn1_model_multi_gpu import *
from Models.dnn2_model import *
from Models.dnn2_model_multi_gpu import *
from Models.preprocessing import *


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--phase', default='train', help='Phase: Can be train, test, gen, val, export, or prod_eval')
  parser.add_argument('--data_location', default='Data', help='Directory to save the tfrecords file in')
  parser.add_argument('--delete_old', default='True', help='Should we keep the old results and tensorboard files')
  parser.add_argument('--model', default='dnn1', help='Model to use: Can be dnn1, dnn2, dnn1_multi, or dnn2_multi')
  parser.add_argument('--num_gpus', default=1, help='select the number of gpus you want youre model to train on')
  parser.add_argument('--file_to_process', default='Data/test.wav', help='this file will be processed during validation')

  args = parser.parse_args()

  configs = {
    'TF_RECORDS_TRAIN_1': os.path.join(args.data_location, 'karaoke_train_1.tfrecords'),
    'TF_RECORDS_TRAIN_2': os.path.join(args.data_location, 'karaoke_train_2.tfrecords'),
    'TF_RECORDS_GEN': os.path.join(args.data_location, 'karaoke_gen.tfrecords'),
    'TF_RECORDS_TEST': os.path.join(args.data_location, 'karaoke_test.tfrecords'),
    'TF_RECORDS_VAL': os.path.join(args.data_location, 'karaoke_val.tfrecords'),
    'TF_RECORDS_META': os.path.join(args.data_location, 'karaoke.meta'),
    'GEN_META_FILE': os.path.join(args.data_location, 'karaoke_gen.meta'),
    'FIRST_DATASET_FILE': 'medleydb_deepkaraoke.json',
    'SECOND_DATASET': 'Data/DSD100/Sources/',
    'TEST_FILE': args.file_to_process,
    'FFT_SIZE': 2048,
    'HOP_SIZE': 512,
    'SAMPLE_RATE': 44100,
    'STACKED_FRAMES': 20,
    'SAMPLE_HOP': 60
  }

  # Train the model
  if args.phase == 'train':
    stats = prepare_train_and_test_data(configs, args)
    model = create_model(args, stats, configs)
    model.train()

  # Test the model
  elif args.phase == 'test':
    stats = prepare_train_and_test_data(configs, args)
    model = create_model(args, stats, configs)
    model.test()

  # generate the training data for dnn2
  elif args.phase == 'gen':
    stats = prepare_train_and_test_data(configs, args)
    _, _, set_one_count, set_two_count = stats
    model = create_model(args, stats, configs)
    model.gen(configs['TF_RECORDS_GEN'], configs['GEN_META_FILE'], set_two_count)

  # validate the model
  elif args.phase == 'val':
    # turn selected track to spectrograms
    stats, spectrograms = prepare_val_data(configs)

    # process with first model
    args.model = 'dnn1_multi'
    first_model = DNN1ModelMultiGpu(args, stats, None)
    combo_matrices, phase_components = first_model.get_second_model_input(spectrograms, configs)

    # process with second model
    args.model = 'dnn2_multi'
    stats = get_stats(configs['GEN_META_FILE'])
    second_model = DNN2ModelMultiGpu(args, stats, None)
    second_model.refine_matrices_and_output(combo_matrices, phase_components, configs, stats)
    print('Validation Done!')

  # Export the model
  elif args.phase == 'export':
    stats = get_stats(configs['TF_RECORDS_META'])
    model = create_model(args, stats)
    model.export()

def create_model(args, stats, configs):
  # Select the model you want to use
  if args.model == "dnn1_multi":
    if args.phase == 'gen':
      model = DNN1ModelMultiGpu(args, stats, configs['TF_RECORDS_TRAIN_2'])
    else:
      model = DNN1ModelMultiGpu(args, stats, configs['TF_RECORDS_TRAIN_1'])

  elif args.model == "dnn1":
    if args.phase == 'gen':
      model = DNN1Model(args, stats, configs['TF_RECORDS_TRAIN_2'])
    else:
      model = DNN1Model(args, stats, configs['TF_RECORDS_TRAIN_1'])

  elif args.model == "dnn2_multi":
    model = DNN2ModelMultiGpu(args, stats, configs['TF_RECORDS_GEN'])

  elif args.model == "dnn2":
    model = DNN2Model(args, stats, configs['TF_RECORDS_GEN'])

  return model

if __name__=="__main__":
  main(sys.argv)
