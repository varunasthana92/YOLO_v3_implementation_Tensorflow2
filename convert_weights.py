#convert_weights.py
import numpy as np
import tensorflow as tf
from yolo_v3 import YOLOv3Net
from yolo_v3 import parse_cfg
def load_weights(model,cfgfile,weightfile):
    # Open the weights file
    fp = open(weightfile, "rb")
    # The first 5 values are header information
    # read and ignore. # The rest of the values are the weights
    np.fromfile(fp, dtype=np.int32, count=5)
    
    # as per the cgf file, block[0] is for [net] which contains the hyperparameters' values
    # hence iterating on block[1:]
    blocks = parse_cfg(cfgfile)
    for i, block in enumerate(blocks[1:]):
        if (block["type"] == "convolutional"):
            conv_layer = model.get_layer('conv_' + str(i+1))
            print("layer: ",i+1,conv_layer)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            if "batch_normalize" in block:
                norm_layer = model.get_layer('bnorm_' + str(i+1))
                print("layer: ",i+1,norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
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
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            # hence need the transformation from [0 1 2 3] --> [2 3 1 0]
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])
    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()
def main():
    weightfile = "weights/yolov3.weights"
    cfgfile = "cfg/yolov3.cfg"
    model_size = (416, 416, 3)
    num_classes = 80
    print('tf version = ', tf.__version__)
    yolo_blocks, output_layers, YOLO_v3_Model = YOLOv3Net(cfgfile, model_size, num_classes)
    YOLO_v3_Model.summary()
    # load_weights(YOLO_v3_Model,cfgfile,weightfile)
    # try:
    #     model.save_weights('weights/yolov3_weights.tf')
    #     print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    # except IOError:
    #     print("Couldn't write the file \'yolov3_weights.tf\'.")
    return

if __name__ == '__main__':
    main()