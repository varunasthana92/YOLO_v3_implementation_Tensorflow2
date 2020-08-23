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
                upsampling_layer = tf.keras.layers.UpSampling2D(stride)(inputs)
                inputs = upsampling_layer(inputs)

        '''
         the attribute layers holds a value of -4 which means that if we are in route block,
         we need to backward 4 layers and then output the feature map from that layer. However,
         for the case of the route block whose attribute 'layers' in cfg file has 2 values like
         in lines 633-634, layers contains -1 and 61, we need to concatenate the feature map
         from a previous layer (-1) and the feature map from layer 61
         '''
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])
            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i

                # "i+start" as we want to move layers[0] backwards from current position and it is already negative
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            # "i+from" as we want to move from[0] backwards from current position and it is already negative
            # and add it to the previous layer feature map
            inputs = outputs[i - 1] + outputs[i + from_]

        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5: num_classes + 5]

            # refine the output by applying sigmoid function to have the value in the range [0,1]
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)

            # instead of using softmax, we use sigmoid for classification
            # as softmax assumes mutually exclusive classes i.e. if it is in classified in one, then cannot
            # be other. Thus in softmax we take the class as the argmax of all the probabilities
            # but with sigmoid, all classes get a probability and are then verified agaisnt a threshold
            # This is useful in more real-world scenario when we have overlapping classes like'dog','animal' 
            classes = tf.sigmoid(classes)
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
    return