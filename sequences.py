from __future__ import absolute_import, print_function

import cv2
import math
import numpy as np
import tensorflow as tf
from utils_image import *
from utils_misc import *
from utils_yolo import *

class TrainSequence(tf.keras.utils.Sequence):
  ''' Every Sequence must implement the __getitem__ and the __len__ methods.
      If you want to modify your dataset between epochs you may implement on_epoch_end.
      The method __getitem__ should return a complete batch.
  '''
  def __init__(self, inputs, targets, batch_size, image_size, letterbox_resize, anchors, num_classes, anchors_mask):
    self.x = inputs
    self.y = targets
    self.batch_size = batch_size
    self.image_size = image_size
    self.letterbox_resize = letterbox_resize
    self.anchors = anchors
    self.num_classes = num_classes
    self.anchors_mask = anchors_mask

  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    # Inputs, read image and scale
    batch_x = self.x[idx*self.batch_size : (idx+1)*self.batch_size]
    batch_y = self.y[idx*self.batch_size : (idx+1)*self.batch_size]
    
    color_table = get_color_table(80)
    classes = get_classes('./data/coco.names')

    inputs = []
    targets = []
      
    for x,y in zip(batch_x, batch_y):
      image = cv2.imread(x) # h,w
      scaled_image, dw, dh, ratio = resize_image(image, self.letterbox_resize, self.image_size)   
      norm_image = normalize_image(scaled_image)
      inputs.append(norm_image)
    
      # Targets, read labels and scale
      y_label = get_file_list(y)
      y_class, y_xywh = parse_targets(y_label) # class, [rx,ry,rw,rh]
  
      y_box = convert_xywh_to_box(y_xywh, image.shape)      
      scaled_box = scale_boxes(y_box, self.letterbox_resize, dw, dh, ratio, 'original')
      box_image_list = convert_box_to_image(scaled_box, y_class, self.image_size, self.num_classes, self.anchors, self.anchors_mask)
      
      targets.append(box_image_list)
      pred_image = draw_boxes(scaled_image, classes, scaled_box, y_class, y_class, color_table)
      #cv2.imshow('Detection result', pred_image)
      #cv2.waitKey(0)
    return np.array(inputs), np.array(targets)
