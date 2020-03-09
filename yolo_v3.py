from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nms import cpu_nms, gpu_nms
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, LeakyReLU, UpSampling2D, ZeroPadding2D

def process_boxes(xy, wh, ratio, grid_size, rescaled_anchors):
  xy = process_xy(xy, ratio, grid_size)
  wh = process_wh(wh, ratio, rescaled_anchors)
  boxes = np.concatenate([xy, wh], axis=-1)

  return boxes

def process_wh(wh, ratio, rescaled_anchors):
  wh = np.exp(wh)*rescaled_anchors
  wh = wh*ratio[::-1]

  return wh

def process_xy(xy, ratio, grid_size):
  xy = 1/(1+np.exp(-xy))

  # Meshgrid for offsets
  grid_x = np.arange(grid_size[1], dtype=int)
  grid_y = np.arange(grid_size[0], dtype=int)
  grid_x, grid_y = np.meshgrid(grid_x, grid_y) # (13,13)
  x_offset = np.expand_dims(grid_x, axis=-1)
  y_offset = np.expand_dims(grid_y, axis=-1)
  xy_offset = np.concatenate([x_offset, y_offset], axis=-1) # (13,13,2)
  xy_offset = np.expand_dims(xy_offset, axis=-2) # (13,13,1,2)
  xy_offset = xy_offset.astype(float)

  xy = xy+xy_offset
  xy = xy*ratio[::-1]

  return xy

def yolo_block(inputs, filters):
  X = conv_layer(inputs, filters, 1)
  X = conv_layer(X, filters*2, 3)
  X = conv_layer(X, filters, 1)
  X = conv_layer(X, filters*2, 3)
  X = conv_layer(X, filters, 1)
  route = X
  X = conv_layer(X, filters*2, 3)
  
  return route, X

def yolo_v3_detector(route_1, route_2, route_3, class_num):
  inter1, X = yolo_block(route_3, 512)
  feature_map_1 = Conv2D(3*(5+class_num), 1, 1, name='feature_map_1')(X)

  inter1 = conv_layer(inter1, 256, 1)
  inter1 = UpSampling2D()(inter1)
  concat1 = Concatenate()([inter1, route_2])

  inter2, X = yolo_block(concat1, 256)
  feature_map_2 = Conv2D(3*(5+class_num), 1, 1, name='feature_map_2')(X)
  
  inter2 = conv_layer(inter2, 128, 1)
  inter2 = UpSampling2D()(inter2)
  concat2 = Concatenate()([inter2, route_1])

  _, X = yolo_block(concat2, 128)
  feature_map_3 = Conv2D(3*(5+class_num), 1, 1, name='feature_map_3')(X)

  return [feature_map_1, feature_map_2, feature_map_3]

def residual_block(inputs, filters):
  shortcut = inputs
  X = conv_layer(inputs, filters, 1)
  X = conv_layer(X, filters*2, 3)
  X = Add()([X, shortcut])

  return X

def conv_layer(input, filters, kernel_size, stride=1, activation=True, batch_norm=True):
  if stride > 1: 
    input = ZeroPadding2D(((1, 0), (1, 0)))(input)

  if stride == 1:
    X = Conv2D(filters, kernel_size, stride, padding='same', use_bias=False)(input)
  else:
    X = Conv2D(filters, kernel_size, stride, padding='valid', use_bias=False)(input)

  if batch_norm:
    X = BatchNormalization()(X)

  if activation:
    X = LeakyReLU(alpha=0.1)(X)
  
  return X

def darknet53(input):
  X = conv_layer(input, 32, 3, 1) # (input, filters, kernel, stride, activation)
  X = conv_layer(X, 64, 3, 2)

  X = residual_block(X, 32)

  X = conv_layer(X, 128, 3, 2)

  for _ in range(2):
    X = residual_block(X, 64)

  X = conv_layer(X, 256, 3, 2)

  for _ in range(8):
    X = residual_block(X, 128)

  route_1 = X
  
  X = conv_layer(X, 512, 3, 2)

  for _ in range(8):
    X = residual_block(X, 256)

  route_2 = X

  X = conv_layer(X, 1024, 3, 2)

  for _ in range(4):
    X = residual_block(X, 512)
 
  route_3 = X

  return route_1, route_2, route_3

class YoloV3():
  def __init__(self, image_size, anchors, num_classes):
    self.feature_extractor = darknet53
    self.detector = yolo_v3_detector
    self.is_training = False
    self.num_classes = num_classes
    self.anchors = anchors
    self.image_size = image_size

  def build_graph(self):
    inputs = tf.keras.Input(self.image_size)
    route_1, route_2, route_3 = self.feature_extractor(inputs)
    outputs = self.detector(route_1, route_2, route_3, self.num_classes)
    self.model = tf.keras.Model(inputs, outputs)
    #self.model.summary()

  def load_weights(self, path):
    self.model.load_weights(path)  

  def predict(self, input):   
    logits_list = self.model.predict(input)
    anchors_list = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]
    
    boxes_list, confs_list, probs_list = [], [], []

    for logits, anchors in zip(logits_list, anchors_list):
      grid_size = list(logits.shape[1:3]) # (13,13), (26,26), (52,52)
      ratio = [x/y for x,y in zip(self.image_size, grid_size)] # (32. 32.)

      rescaled_anchors = [(anchor[0]/ratio[1], anchor[1]/ratio[0]) for anchor in anchors]
      logits = np.reshape(logits, [-1, grid_size[0], grid_size[1], 3, 5+self.num_classes])

      xy, wh, conf, prob = np.split(logits, [2, 4, 5], axis=-1) # [0,1], [2,3], [4], [5...]
  
      boxes = process_boxes(xy, wh, ratio, grid_size, rescaled_anchors)
      boxes = np.reshape(boxes, [-1, grid_size[0]*grid_size[1]*3, 4])

      conf = np.reshape(conf, [-1, grid_size[0]*grid_size[1]*3, 1])
      conf = 1/(1+np.exp(-conf))

      prob = np.reshape(prob, [-1, grid_size[0]*grid_size[1]*3, self.num_classes])      
      prob = 1/(1+np.exp(-prob))

      boxes_list.append(boxes)
      confs_list.append(conf)
      probs_list.append(prob)

    boxes = np.concatenate(boxes_list, axis=1)
    confs = np.concatenate(confs_list, axis=1)
    probs = np.concatenate(probs_list, axis=1)

    center_x, center_y, width, height = np.split(boxes, [1, 2, 3], axis=-1)
    x_min = center_x-width/2
    y_min = center_y-height/2
    x_max = center_x+width/2
    y_max = center_y+height/2

    boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=-1)
    scores = confs*probs

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    boxes, scores, labels = cpu_nms(boxes, scores, self.num_classes, max_boxes=200, score_thresh=0.3, iou_thresh=0.5)

    return boxes, scores, labels
