from __future__ import absolute_import
from __future__ import print_function

import argparse
#import cv2
#from utils_image import *
from utils_yolo import *
import numpy as np
#import tensorflow as tf
from model_trainer import ModelTrainer
from yolo_v3 import YoloV3

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_input', type=str, default='./coco/trainvalno5k.image.txt')
  parser.add_argument('--train_target', type=str, default='./coco/trainvalno5k.label.txt')
  parser.add_argument('--dev_input', type=str, default='/scratch/chsu/coco/5k.image.txt')
  parser.add_argument('--dev_target', type=str, default='/scratch/chsu/coco/5k.label.txt')
  parser.add_argument('--anchor_path', type=str, default='./data/yolo_anchors.txt')
  parser.add_argument('--class_name_path', type=str, default='./data/coco.names')
  parser.add_argument('--image_shape', nargs='+', default=[416,416,3])
  parser.add_argument('--letterbox_resize', action='store_true', default=True)

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
  anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

  trainer = ModelTrainer()
  trainer.get_input_options(args.image_shape, args.letterbox_resize, anchors, num_classes, anchors_mask)
  trainer.get_hyperparameters(args.num_epochs, args.batch_size)
  trainer.get_train_set(args.train_input, args.train_target) 
  #dev_set = get_train_set(args.dev_set)

 
  #model.load_weights(args.weights)
  # Train Model
