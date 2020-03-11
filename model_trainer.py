from __future__ import absolute_import, print_function

#from callbacks import *
#from data_io import write_history
from sequences import TrainSequence#, DevSequence

#import csv
#import tensorflow as tf

class ModelTrainer():
  def __init__(self):
    self.fit_options = {}

  def get_hyperparameters(self, num_epochs, batch_size):
    self.fit_options['epochs'] = num_epochs
    self.fit_options['batch_size'] = batch_size
    #self.fit_options['verbose'] = int(hp_opts.verbose)

  def get_train_set(self, image_path, label_path):
    image_list = get_file_list(image_path)
    label_list = get_file_list(label_path)
    
    train_set = TrainSequence(image_list, label_list, self.fit_options['batch_size'])
    self.fit_options['x'] = train_set

    num_samples = len(inputs[0])
    print('Number of Training Samples: ', num_samples)

    num_sequences = len(inputs[0][0])
    print('Number of Input Sequences: ', num_sequences)

def get_file_list(file_path):
  with open(file_path, 'r') as f:
    file_list = f.readlines()
  return file_list
