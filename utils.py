#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import cv2
import os
import random
import colorsys

def resize_image(inputs, modelsize):
    inputs= tf.image.resize(inputs, (modelsize[0], modelsize[1]))
    inputs = inputs /255
    return inputs

def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    num_classes = len(class_names)
    hsv_tuples = [(2.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # set seed to have same random output on each call
    # thus have same colors assigned to respective classes on each run of the program
    random.seed(7)
    random.shuffle(colors)
    random.seed(None)

    for i in range(nums):
        class_ind = int(classes[i])
        box_color = colors[class_ind]
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), box_color, 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[class_ind], objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, box_color, 2)
    return img

def SetGPU(GPUNum=-1):
    if(GPUNum != -1):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUNum)