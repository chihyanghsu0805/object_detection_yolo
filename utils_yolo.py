from __future__ import absolute_import, print_function

import numpy as np

def convert_box_to_image(boxes, labels, image_size, num_classes, anchors, anchors_mask):
  # convert boxes form:
  # shape: [N, 2]
  # (x_center, y_center)
  xy = (boxes[:, 0:2]+boxes[:, 2:4])/2
  # (width, height)
  wh = boxes[:, 2:4]-boxes[:, 0:2]

  # [13, 13, 3, 5+num_class+1] `5` means coords and labels.
  y_true_13 = np.zeros((image_size[1] // 32, image_size[0] // 32, 3, 5+num_classes), np.float32)
  y_true_26 = np.zeros((image_size[1] // 16, image_size[0] // 16, 3, 5+num_classes), np.float32)
  y_true_52 = np.zeros((image_size[1] // 8, image_size[0] // 8, 3, 5+num_classes), np.float32)

  # [N, 1, 2]
  wh = np.expand_dims(wh, 1)
  # broadcast tricks
  # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
  mins = np.maximum(-wh/2, -anchors/2)
  maxs = np.minimum(wh/2, wh/2)
  # [N, 9, 2]
  whs = maxs-mins

  # [N, 9]
  intersection = whs[:,:,0]*whs[:,:,1]
  union = wh[:,:,0]*wh[:,:,1]+anchors[:,0]*anchors[:,1]-whs[:,:,0]*whs[:,:,1]+1e-10
  iou =  intersection/union
  # [N]
  best_match_idx = np.argmax(iou, axis=1)

  for idx, _xy, _wh, label in zip(best_match_idx, xy, wh, labels):
    
    if idx in [0, 1, 2]:
      ratio = 8
      x = int(np.floor(_xy[0]/ratio))
      y = int(np.floor(_xy[1]/ratio))        

      k = [0,1,2].index(idx)
      y_true_52[y, x, k, :2] = _xy
      y_true_52[y, x, k, 2:4] = _wh
      y_true_52[y, x, k, 4] = 1.
      y_true_52[y, x, k, 5+label] = 1.        

    if idx in [3, 4, 5]:
      ratio = 16
      x = int(np.floor(_xy[0]/ratio))
      y = int(np.floor(_xy[1]/ratio))        

      k = [3,4,5].index(idx)
      y_true_26[y, x, k, :2] = _xy
      y_true_26[y, x, k, 2:4] = _wh
      y_true_26[y, x, k, 4] = 1.
      y_true_26[y, x, k, 5+label] = 1.        
        
    if idx in [6, 7, 8]:
      ratio = 32
      x = int(np.floor(_xy[0]/ratio))
      y = int(np.floor(_xy[1]/ratio))        

      k = [6,7,8].index(idx)
      y_true_13[y, x, k, :2] = _xy
      y_true_13[y, x, k, 2:4] = _wh
      y_true_13[y, x, k, 4] = 1.
      y_true_13[y, x, k, 5+label] = 1.        

  y_true = [y_true_13, y_true_26, y_true_52]

  return y_true

def convert_xywh_to_box(xywh, image_shape):
  x0 = xywh[:,0]*image_shape[1]
  y0 = xywh[:,1]*image_shape[0]
  w0 = xywh[:,2]*image_shape[1]
  h0 = xywh[:,3]*image_shape[0]
      
  xywh[:,0] = x0-w0/2
  xywh[:,1] = y0-h0/2
  xywh[:,2] = x0+w0/2
  xywh[:,3] = y0+h0/2
  
  box = xywh
  return box 

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
