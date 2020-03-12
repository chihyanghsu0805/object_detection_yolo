from __future__ import absolute_import, print_function

import cv2
import numpy as np
import random

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

def draw_boxes(image, classes, boxes, scores, labels, color_table):
  for box,score,label in zip(boxes, scores, labels):
    plot_one_box(image, box, label=classes[label] + ', {:.2f}%'.format(score*100), color=color_table[label])

  return image

def scale_coordinates(boxes, bool_letterbox_resize, dw, dh, ratio):
  boxes[:, [0,2]] = (boxes[:, [0,2]]-dw)/ratio[1]
  boxes[:, [1,3]] = (boxes[:, [1,3]]-dh)/ratio[0]
  #if bool_letterbox_resize:
  #  boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
  #  boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
  #else:
  #  boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
  #  boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

  return boxes

def normalize_image(re_image):
  re_image = cv2.cvtColor(re_image, cv2.COLOR_BGR2RGB)
  re_image = np.asarray(re_image, np.float32)
  re_image = re_image[np.newaxis, :] / 255.

  return re_image

def resize_image(image, bool_letterbox_resize, new_size):
  if bool_letterbox_resize:
    image, resize_ratio, dw, dh = letterbox_resize(image, new_size)

  else:
    old_size = image.shape[:2]
    image = cv2.resize(image, tuple(new_size))
    dw, dh = 0
    resize_ratio = [x/y for x,y in zip(new_size, old_size)]

  return image, dw, dh, resize_ratio
  
def plot_one_box(image, coord, label=None, color=None, line_thickness=None):
  ''' 
  coord: [x_min, y_min, x_max, y_max] format coordinates.
  image: image to plot on.
  label: str. The label name.
  color: int. color index.
  line_thickness: int. rectangle line thickness.
  '''
  tl = line_thickness or int(round(0.002 * max(image.shape[0:2])))  # line thickness
  color = color or [random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
  cv2.rectangle(image, c1, c2, color, thickness=tl)
  if label:
    tf = max(tl-1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=float(tl)/3, thickness=tf)[0]
    c2 = c1[0]+t_size[0], c1[1]-t_size[1]-3
    cv2.rectangle(image, c1, c2, color, -1)  # filled
    cv2.putText(image, label, (c1[0], c1[1]-2), 0, float(tl)/3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def letterbox_resize(image, new_size, interp=0):
  ''' 
  Letterbox resize. keep the original aspect ratio in the resized image.
  '''
  old_size = image.shape[:2]

  ratio_list = [x/y for (x,y) in zip(new_size,old_size)]
  resize_ratio = min(ratio_list)

  resize_w = int(resize_ratio*old_size[1])
  resize_h = int(resize_ratio*old_size[0])

  image = cv2.resize(image, (resize_w, resize_h), interpolation=interp)
  image_padded = np.full((new_size[1], new_size[0], 3), 128, np.uint8)

  dw = int((new_size[1]-resize_w)/2)
  dh = int((new_size[0]-resize_h)/2)

  image_padded[dh: resize_h+dh, dw: resize_w+dw, :] = image
  ratio_list = [resize_ratio for _ in ratio_list]

  return image_padded, ratio_list, dw, dh

