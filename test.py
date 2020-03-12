from __future__ import absolute_import
from __future__ import print_function

import argparse
import cv2
from utils_image import *
from utils_yolo import *
import numpy as np

from yolo_v3 import YoloV3

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", type=str, required=True)
  parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt")
  parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416])
  parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True)
  parser.add_argument("--class_name_path", type=str, default="./data/coco.names")
  parser.add_argument("--weights", type=str, default="./yolo_tf_weights/yolov3.weights.h5")
  args = parser.parse_args()

  anchors = get_anchors(args.anchor_path)
  print('Anchors:', anchors)
  classes = get_classes(args.class_name_path)

  num_classes = len(classes)
  print('Number of Classes', num_classes)

  color_table = get_color_table(num_classes)

  raw_image = cv2.imread(args.image)

  image, dw, dh, ratio = resize_image(raw_image, args.letterbox_resize, args.new_size)
  image = normalize_image(image)
  
  image_shape = list(np.squeeze(image).shape)
  print('Image shape:', image_shape)

  model = YoloV3(image_shape, anchors, num_classes)
  model.build_graph()
 
  model.load_weights(args.weights)
  pred_boxes, pred_scores, pred_labels = model.predict(image)

  # TODO: Save Outcome
  print(pred_boxes)
  pred_boxes = scale_coordinates(pred_boxes, args.letterbox_resize, dw, dh, ratio)

  print("Box Coordinates:")
  print(pred_boxes)

  print("Box Scores:")
  print(pred_scores)

  print("Box Labels:")
  print(pred_labels)
  
  pred_image = draw_boxes(raw_image, classes, pred_boxes, pred_scores, pred_labels, color_table)

  cv2.imshow('Detection result', pred_image)
  cv2.imwrite('detection_result.jpg', pred_image)
  cv2.waitKey(0)


