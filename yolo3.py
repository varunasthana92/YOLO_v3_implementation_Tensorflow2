#! /usr/bin/env python
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D
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


def YOLOv3Net(cfgfile, imgPH, model_size, num_classes):
    blocks = parse_cfg(cfgfile)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    input_image = tf.keras.Input(shape=model_size)
    inputs = input_image
    inputs = inputs / 255.0
    # inputs = input_image=imgPH

    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    for i, block in enumerate(blocks[1:]):
        print('Layer num i = ',i)
        if block["type"] == "convolutional":
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                pad_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
                inputs = pad_layer(inputs)

            # in_shape = inputs.shape.as_list()
            conv2D_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                            padding='valid' if strides > 1 else 'same', name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True) # input_shape=in_shape)
            
            inputs = conv2D_layer(inputs)
            if "batch_normalize" in block:
                inputs = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i))(inputs)
                inputs = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsampling2D_layer = tf.keras.layers.UpSampling2D(stride)
            inputs = upsampling2D_layer(inputs)

            '''
             the attribute layers holds a value of -4 which means that if we are in route block,
             we need to backward 4 layers and then output the feature map from that layer. However,
             for the case of the route block whose attribute 'layers' in cfg file has 2 values like
             in lines 633-634, layers contains -1 and 61, we need to concatenate the feature map
             from a previous layer (-1) and the feature map from layer 61
             '''
        elif block["type"] == "route":
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
        outputs[i] = inputs
        output_filters.append(filters)
    model = Model(input_image, out_pred)
    # model.summary()
    return model
    # return out_pred


def YOLOv3Net_pre_trained(cfgfile, imgPH, model_size, num_classes, weightfile):
    blocks = parse_cfg(cfgfile)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    # input_image = tf.keras.Input(shape=model_size)
    # inputs = input_image
    # inputs = inputs / 255.0
    inputs = input_image = imgPH

    fp = open(weightfile, "rb")
    # The first 5 values are header information
    # read and ignore
    np.fromfile(fp, dtype=np.int32, count=5)

    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    for i, block in enumerate(blocks[1:]):
        print('Layer num i = ', i)
        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"]) # number of filteres or kernels
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                pad_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
                inputs = pad_layer(inputs)

            in_shape = inputs.shape.as_list()
            conv2D_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                            padding='valid' if strides > 1 else 'same', name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True, 
                            input_shape=in_shape)
            # num_filters = conv2D_layer.filters    
            # k_size = conv2D_layer.kernel_size[0]
            # in_dim = conv2D_layer.input_shape[-1]
            in_dim = in_shape[-1]
            norm_layer = None
            bn_weights = None
            conv_bias = None
            if "batch_normalize" in block:
                norm_layer = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i))
                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(fp, dtype=np.float32, count=size * filters)
                # from the weight file the data is in format [gamma, beta, mean, variance]
                # tf batch_norm has [beta, gamma, mean, variance]
                # hence need to do the [0 1 2 3] --> [1 0 2 3]

                # all attributes of wieghts are stacked for each kernel in rows
                # thus weights = [ [beta1,  beta2,  beta3,  ....beta_n  ],
                #                  [gamm1,  gamma2, gamma3, ....gamma_n ],
                #                  [mean1,  mean2,  mean3,  ....mean_n  ],
                #                  [var1,   var2,   var3    ....var_n   ]]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, kernel_size, kernel_size) # layer weight shape
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            # hence need the transformation from [0 1 2 3] --> [2 3 1 0]
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv2D_layer.set_weights([conv_weights])
                inputs = conv2D_layer(inputs)
                inputs = norm_layer(inputs)
                inputs = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)
                print("layer: ", i + 1, conv2D_layer)
                print("layer: ", i + 1, norm_layer)
            else:
                conv2D_layer.set_weights([conv_weights, conv_bias])
                inputs = conv2D_layer(inputs)
                print("layer: ", i + 1, conv2D_layer)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsampling2d_layer = tf.keras.layers.UpSampling2D(stride)(inputs)
            inputs = upsampling2d_layer(inputs)

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
            strides = (input_image.shape[1] // out_shape[1], input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)

    # model = Model(input_image, out_pred)
    # model.summary()
    # return model
    return out_pred