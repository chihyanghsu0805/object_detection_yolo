from __future__ import absolute_import
from __future__ import print_function

import argparse
#import cv2
#from utils_image import *
import numpy as np
#import tensorflow as tf
from model_trainer import ModelTrainer
from yolo_v3 import YoloV3

def get_train_set(file_path):
  with open(file_path, 'r') as f:
    x = f.readlines()
  return x

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_set', type=str, default='/scratch/chsu/coco/trainvalno5k.txt')
  parser.add_argument('--dev_set', type=str, default='/scratch/chsu/coco/5k.txt')
  parser.add_argument('--anchor_path', type=str, default='./data/yolo_anchors.txt')
  parser.add_argument('--class_name_path', type=str, default='./data/coco.names')
  parser.add_argument('--image_shape', nargs='+', default=[416,416,3])
  #parser.add_argument('--weights', type=str, default='./yolo_tf_weights/yolov3.weights.h5')
  parser.add_argument('--num_epochs', type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=32)
  args = parser.parse_args()

  anchors = get_anchors(args.anchor_path)
  print('Anchors:', anchors)

  classes = get_classes(args.class_name_path)
  num_classes = len(classes)
  print('Number of Classes:', num_classes)
  
  print('Image Shape', args.image_shape)
  model = YoloV3(args.image_shape, anchors, num_classes)
  model.build_graph()
  #model.compile()
  
  # Dataset
  # Label: class, x, y, w, h (relative to image)
  trainer = ModelTrainer()
  trainer.get_hyperparameters(args.num_epochs, args.batch_size)
  trainer.get_train_set(args.train_set) 
  #dev_set = get_train_set(args.dev_set)

 
  #model.load_weights(args.weights)
  # Train Model
