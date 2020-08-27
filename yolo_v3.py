#! /usr/bin/env python
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

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
    outputs = []
    # output_filters = []
    filters = []
    # out_pred = []
    yolo_blocks = []
    output_layers = []
    # scale = 0
    input_image = tf.keras.Input(shape=model_size)
    inputs = input_image
    yolo_reached = False
    # inputs = inputs / 255.0

    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    for i, block in enumerate(blocks[1:]):
        # print('Layer num i = ',i, 'with block = ', block['type' ])
        if block["type"] == "convolutional":
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                pad_layer = tf.keras.layers.ZeroPadding2D(int(kernel_size//2), name='pad_' + str(i+1))
                inputs = pad_layer(inputs)

            # in_shape = inputs.shape.as_list()
            conv2D_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                            padding='valid' if strides > 1 else 'same', name='conv_' + str(i+1),
                            use_bias=False if (activation == 'leaky') else True)
            
            inputs = conv2D_layer(inputs)
            if activation == 'leaky':
                inputs = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i+1))(inputs)
                inputs = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i+1))(inputs)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsampling2D_layer = tf.keras.layers.UpSampling2D(stride, name='upsample_' + str(i+1))
            inputs = upsampling2D_layer(inputs)

        elif block["type"] == "route":
            '''
             the attribute 'layers' holds a value of -4 which means that if we are in route block,
             we need to backward 4 layers in the cfg file and then output the feature map from that
             layer. However,for the case of the route block whose attribute 'layers' in cfg file has
             2 values like in lines 633-634, layers contains -1 and 61, we need to concatenate the
             feature map from a previous layer (-1) and the feature map from layer 61
             '''
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])
            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                # For ex: start = -1, and the network have 4 blocks executed, and 5th block (current)
                # is 'route' block. Thus we need "inputs = 4rd block output" i.e. 5-1
                # For index correction by '-1', inputs = idx[3] block output
                # Also since out 'output_filter' variable is behind the current position by 1
                # that is output_filter has data upto 4th block (or idx[3] block), then
                # idx[3] = idx[-1] = idx[staart], thus inputs = output_filtes[start]
                
                # filters = output_filters[start] + output_filters[end - 1]  # end-1 is idx correction
                inputs = tf.keras.layers.concatenate([outputs[start], outputs[end-1]], axis=-1, name='route_' + str(i+1))
            else:
                # filters = output_filters[1 + start]
                inputs = outputs[start]
                if yolo_reached:
                    shape = inputs.shape.as_list()
                    temp = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]), name = 'dummy_yolo_reshape_'+ str(i))(outputs[-1])
                    temp = tf.keras.layers.Lambda(lambda x: x*0, name = 'dummy_yolo_all_zero_'+ str(i))(temp)
                    inputs = tf.keras.layers.add([inputs, temp], name = 'route_' + str(i+1))
                    yolo_reached =  False

        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            # "1+from" as we want to move from[0] backwards from current position and it is already negative
            # and add it to the previous layer feature map i.e outputs[-1]
            inputs = tf.keras.layers.add([outputs[-1], outputs[from_]], name='shortcut_' + str(i+1))
            
        elif block["type"] == "yolo":
            yolo_blocks.append(block)
            output_layers.append(outputs[-1])
            # inputs = tf.keras.layers.Lambda(lambda x: x*0, name = 'yolo_zero_'+ str(i+1))(inputs)
            yolo_reached = True

            '''            
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()
            # inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])
            # box_centers = inputs[:, :, 0:2]
            # box_shapes = inputs[:, :, 2:4]
            # confidence = inputs[:, :, 4:5]
            # classes = inputs[:, :, 5: num_classes + 5]
            model_out = inputs
            inputs_curr = tf.reshape(model_out, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])
            box_centers = inputs_curr[:, :, 0:2]
            box_shapes = inputs_curr[:, :, 2:4]
            confidence = inputs_curr[:, :, 4:5]
            classes = inputs_curr[:, :, 5: num_classes + 5]

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

            # convert relative positions of the center boxes into the real positions i.e. use the formulation
            # given by the author in the original paper for bx, by
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)
            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])
            in_shape = input_image.shape.as_list()
            strides = (in_shape[1] / out_shape[1], in_shape[2] / out_shape[2])
            print(input_image.shape)
            print(out_shape[2])
            print('strides = ', strides)
            print('box_before shape', box_centers.shape)
            box_centers = tf.multiply((box_centers + cxy), strides)
            # print('box after shape', box_centers.shape)
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
            '''
        outputs.append(inputs)
    YOLO_v3_Model = Model(input_image, inputs)
    return yolo_blocks, output_layers, YOLO_v3_Model