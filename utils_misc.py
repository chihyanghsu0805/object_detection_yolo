from __future__ import absolute_import, print_function

import numpy as np

def parse_targets(y_label):
  y_class = []
  y_box = []
  
  for i in y_label:
  	i_list = i.split(' ')
  	y_class.append(int(i_list[0]))
  	y_box.append([float(j) for j in i_list[1:5]])
  
  return np.array(y_class), np.array(y_box)

def get_file_list(file_path):
  with open(file_path, 'r') as f:
    file_list = f.read().splitlines()
  return file_list