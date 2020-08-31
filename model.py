#! /usr/bin/env python
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines   = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    
    holder      = {}
    blocks      = []
    
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}    
        key, value              = line.split("=")
        holder[key.rstrip()]    = value.lstrip()
    
    blocks.append(holder)
    return blocks


def YOLOv3Net(cfgfile, num_classes= 80, model_size = (416, 416, 3), max_total_size= 100,\
              max_output_size_per_class=100, iou_threshold= 0.5, score_threshold= 0.5):
    blocks              = parse_cfg(cfgfile)
    output_each_layer   = []
    scale               = 1
    inputs              = input_img=tf.keras.Input(shape=model_size)
    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    conv_count          = 0
    norm_count          = 0
    for i, block in enumerate(blocks[1:]):
        if block["type"] == "convolutional":
            activation      = block["activation"]
            filters         = int(block["filters"])
            kernel_size     = int(block["size"])
            strides         = int(block["stride"])
            if strides > 1:
                pad_layer   = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='pad_' + str(i+1))
                inputs      = pad_layer(inputs)

            conv2D_layer    = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                    padding='valid' if strides > 1 else 'same', name='conv2d_' + str(conv_count),
                                    use_bias=False if (activation == 'leaky') else True)
            conv_count     +=1
            inputs          = conv2D_layer(inputs)
            
            if activation == 'leaky':
                inputs      = tf.keras.layers.BatchNormalization(name='bnorm_' + str(norm_count))(inputs)
                norm_count +=1
                inputs      = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i+1))(inputs)

        elif block["type"] == "upsample":
            stride              = int(block["stride"])
            upsampling2D_layer  = tf.keras.layers.UpSampling2D(stride, name='upsample_' + str(i+1))
            inputs              = upsampling2D_layer(inputs)

        elif block["type"] == "route":
            '''
             the attribute 'layers' holds a value of -4 which means that if we are in route block,
             we need to move backward 4 layers in the cfg file and use the feature map from that
             layer. However,for the case of the route block whose attribute 'layers' in cfg file has
             2 values like in lines 633-634, layers contains -1 and 61, we need to concatenate the
             feature map from a previous layer (-1) and the feature map from layer 61
             '''
            block["layers"]     = block["layers"].split(',')
            start               = int(block["layers"][0])
            
            if len(block["layers"]) > 1:
                end     = int(block["layers"][1])
                # For ex: start = -1, and the network have 4 blocks executed, and 5th block (current)
                # is 'route' block. Thus we need "inputs = 4rd block output" i.e. 5-1
                # For index correction by '-1', inputs = idx[3] block output
                # Also since out 'output_filter' variable is behind the current position by 1
                # that is output_filter has data upto 4th block (or idx[3] block), then
                # idx[3] = idx[-1] = idx[staart], thus inputs = output_filtes[start]
                inputs  = tf.keras.layers.concatenate([output_each_layer[start], output_each_layer[end]], axis=-1, name='route_' + str(i+1))
            else:
                inputs  = output_each_layer[start]

        elif block["type"] == "shortcut":
            from_       = int(block["from"])
            # "1+from" as we want to move from[0] backwards from current position and it is already negative
            # and add it to the previous layer feature map i.e output_each_layer[-1]
            inputs      = tf.keras.layers.add([output_each_layer[-1], output_each_layer[from_]], name='shortcut_' + str(i+1))
            
        elif block["type"] == "yolo":
            inputs = decode(inputs, block, model_size, num_classes)
            if scale == 1:
                boxes_0 = inputs
                scale  +=1
            elif scale == 2:
                boxes_1 = inputs
                scale  +=1
            elif scale == 3:
                boxes_2 = inputs
                scale  +=1

        output_each_layer.append(inputs)
    best_boxes      = get_box_nms([boxes_0, boxes_1, boxes_2], max_total_size, max_output_size_per_class,\
                                   iou_threshold, score_threshold)
    YOLO_v3_Model   = Model(input_img, outputs = best_boxes)
    return YOLO_v3_Model


def get_box_nms(preds, max_total_size= 100, max_output_size_per_class=100, iou_threshold= 0.5, score_threshold= 0.5):
    bbox        = []
    objectness  = []
    class_prob  = []
    for pred in preds:
        box     = pred[0]
        obj     = pred[1]
        prob    = pred[2]
        bbox.append(tf.reshape(box, (tf.shape(box)[0], -1, tf.shape(box)[-1])))
        objectness.append(tf.reshape(obj, (tf.shape(obj)[0], -1, tf.shape(obj)[-1])))
        class_prob.append(tf.reshape(prob, (tf.shape(prob)[0], -1, tf.shape(prob)[-1])))        

    bbox        = tf.concat([bbox[0], bbox[1], bbox[2]], axis=-2)
    objectness  = tf.concat([objectness[0], objectness[1], objectness[2]], axis=-2)
    class_prob  = tf.concat([class_prob[0], class_prob[1], class_prob[2]], axis=-2)
    scores      = objectness * class_prob
    
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                                                    boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
                                                    scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
                                                    max_output_size_per_class = max_output_size_per_class,
                                                    max_total_size = max_total_size,
                                                    iou_threshold = iou_threshold,
                                                    score_threshold = score_threshold
                                                )

    return boxes, scores, classes, valid_detections

def decode(inputs, block, model_size, num_classes):                
    mask            = block["mask"].split(",")
    mask            = [int(x) for x in mask]
    anchors         = block["anchors"].split(",")
    anchors         = [int(a) for a in anchors]
    anchors         = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors         = np.array([anchors[k] for k in mask]) * 1. / model_size[0]
    n_anchors       = len(anchors)
    out_shape       = inputs.get_shape().as_list()
    batch_size      =  1 if out_shape[0] == None else out_shape[0]
    
    inputs          = tf.reshape(inputs, (-1, out_shape[1], out_shape[2], n_anchors, 5 + num_classes))
    conv_raw_dxdy   = inputs[:, : , :, : , 0:2]
    conv_raw_dwdh   = inputs[:, : , :, : , 2:4]
    conv_raw_conf   = inputs[:, : , :, : , 4:5]
    conv_raw_prob   = inputs[:, : , :, : , 5: num_classes + 5]

    y               = tf.tile(tf.range(out_shape[1], dtype=tf.int32)[:, tf.newaxis], [1, out_shape[2]])
    x               = tf.tile(tf.range(out_shape[2], dtype=tf.int32)[tf.newaxis, :], [out_shape[1], 1])
    cx_cy_grid      = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    cx_cy_grid      = tf.tile(cx_cy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, n_anchors, 1])
    cx_cy_grid      = tf.cast(cx_cy_grid, tf.float32)
    
    # refine the output by applying sigmoid function to have the value in the range [0,1]
    # convert relative positions of the center boxes into the real positions i.e. use the formulation
    # given by the author in the original paper for bx, by
    pred_xy         = (tf.sigmoid(conv_raw_dxdy) + cx_cy_grid) / out_shape[1:3]
    pred_wh         = (tf.exp(conv_raw_dwdh) * anchors)

    pred_x1y1       = pred_xy - pred_wh / 2
    pred_x2y2       = pred_xy + pred_wh / 2

    pred_xy1xy2     = tf.concat([pred_x1y1, pred_x2y2], axis=-1)

    # instead of using softmax, we use sigmoid for classification
    # as softmax assumes mutually exclusive classes i.e. if it is in classified in one, then cannot
    # be other. Thus in softmax we take the class as the argmax of all the probabilities
    # but with sigmoid, all classes get a probability and are then verified agaisnt a threshold
    # This is useful in more real-world scenario when we have overlapping classes like'dog','animal' 
    pred_conf       = tf.sigmoid(conv_raw_conf)
    pred_prob       = tf.sigmoid(conv_raw_prob)
    return pred_xy1xy2, pred_conf, pred_prob