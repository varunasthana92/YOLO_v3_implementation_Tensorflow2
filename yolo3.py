#! /usr/bin/env python
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D


def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks


def YOLOv3Net(cfgfile, model_size, num_classes):
    blocks = parse_cfg(cfgfile)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    input_layer = tf.keras.Input(shape=model_size)
    inputs = input_layer
    inputs = inputs / 255.0

    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    for i, block in enumerate(blocks[1:]):
        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                pad_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
                inputs = pad_layer(inputs)

            conv2D_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                            padding='valid' if strides > 1 else 'same', name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True)
            
            inputs = conv2D_layer(inputs)
            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif (block["type"] == "upsample"):
                stride = int(block["stride"])
                inputs = UpSampling2D(stride)(inputs)
    return

input_layer = tf.keras.Input(shape= (416,416))