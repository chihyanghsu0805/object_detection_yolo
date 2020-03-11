from __future__ import absolute_import, print_function

import math
import numpy as np
import tensorflow as tf

class TrainSequence(tf.keras.utils.Sequence):
  ''' Every Sequence must implement the __getitem__ and the __len__ methods.
      If you want to modify your dataset between epochs you may implement on_epoch_end.
      The method __getitem__ should return a complete batch.
  '''
  def __init__(self, inputs, targets, batch_size):
    self.x = inputs
    self.y = targets
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    # Inputs, read image and scale
    batch_x = self.x[idx*self.batch_size : (idx+1)*self.batch_size]
    inputs = []
    for x in batch_x:
      input = read_nii_from_list(x)
      inputs.append(cat_input)
    
    # Targets, read labels and scale
    targets = get_targets(self.y, idx, self.batch_size)

    return np.array(inputs), targets
