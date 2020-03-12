from __future__ import absolute_import, print_function

import numpy as np

def get_classes(class_name_path):
  names = {}
  with open(class_name_path, 'r') as data:
    for ID, name in enumerate(data):
      names[ID] = name.strip('\n')
  return names

def get_anchors(anchor_path):
  ''' 
  parse anchors.
  returned data: shape [N, 2], dtype float32
  '''
  anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
  return anchors