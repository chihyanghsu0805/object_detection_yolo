from __future__ import absolute_import, print_function

#from callbacks import *
#from data_io import write_history
from sequences import TrainSequence#, DevSequence
from utils_misc import *
#import csv
#import tensorflow as tf

class ModelTrainer():
  def __init__(self):
    self.fit_options = {}
    self.input_options = {}

  def get_input_options(self, image_size, letterbox_resize, anchors, num_classes, anchors_mask):
    self.input_options['image_size'] = image_size
    self.input_options['letterbox_resize'] = letterbox_resize
    self.input_options['anchors'] = anchors
    self.input_options['num_classes'] = num_classes
    self.input_options['anchors_mask'] = anchors_mask

  def get_hyperparameters(self, num_epochs, batch_size):
    self.fit_options['epochs'] = num_epochs
    self.fit_options['batch_size'] = batch_size
    #self.fit_options['verbose'] = int(hp_opts.verbose)

  def get_train_set(self, image_path, label_path):
    image_list = get_file_list(image_path)
    label_list = get_file_list(label_path)
    train_set = TrainSequence(image_list, label_list, self.fit_options['batch_size'], **self.input_options)
    self.fit_options['x'] = train_set
    
    num_samples = len(image_list)
    print('Number of Training Samples: ', num_samples)