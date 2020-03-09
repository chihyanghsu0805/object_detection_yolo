from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

def res_block(inputs, filters):
  shortcut = inputs
  x = conv2d(inputs, filters*1, 1)
  x = conv2d(x, filters*2, 3)
  x = x + shortcut

  return net

def fixed_padding(inputs, kernel_size):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode='CONSTANT')

  return padded_inputs

def conv2d(inputs, filters, kernel_size, strides=1):

  if strides > 1: 
    inputs = fixed_padding(inputs, kernel_size)

  if strides == 1:
    inputs = tf.keras.layers.Conv2D(filters, kernel_size, stride=strides, padding='same', activation='relu')(inputs)
  else:
    inputs = tf.keras.layers.Conv2D(filters, kernel_size, stride=strides, padding='valid', activation='relu')(inputs)

  return inputs

def darknet(inputs):
  x = conv2d(inputs, 32,  3, strides=1)
  x = conv2d(x, 64,  3, strides=2)

  # res_block * 1
  x = res_block(x, 32)

  x = conv2d(x, 128, 3, strides=2)

  # res_block * 2
  for _ in range(2):
    x = res_block(x, 64)

  x = conv2d(x, 256, 3, strides=2)

  # res_block * 8
  for _ in range(8):
    x = res_block(x, 128)

  route_1 = x
 
  x = conv2d(x, 512, 3, strides=2)

  # res_block * 8
  for _ in range(8):
    x = res_block(x, 256)

  route_2 = x

  x = conv2d(x, 1024, 3, strides=2)

  # res_block * 4
  for _ in range(4):
    x = res_block(x, 512)
 
  route_3 = x

  return route_1, route_2, route_3

def yolo_block(inputs, filters):
  x = conv2d(inputs, filters*1, 1)
  x = conv2d(x, filters*2, 3)
  x = conv2d(x, filters*1, 1)
  x = conv2d(x, filters*2, 3)
  x = conv2d(x, filters*1, 1)
  route = x
  x = conv2d(x, filters*2, 3)

  return route, x
