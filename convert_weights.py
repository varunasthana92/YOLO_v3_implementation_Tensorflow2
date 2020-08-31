#convert_weights.py
import numpy as np
import tensorflow as tf
from model import YOLOv3Net, parse_cfg
import argparse

def load_weights(model, cfgfile, weightfile):
    # Open the pre-trained weights file
    fp = open(weightfile, "rb")
    # The first 5 values are header information
    # read and ignore. # The rest of the values are the weights
    np.fromfile(fp, dtype=np.int32, count=5)
    blocks = parse_cfg(cfgfile)
    norm_count = 0
    conv_count = 0
    for i, block in enumerate(blocks[1:]):
        try:
            if (block["type"] == "convolutional"):
                conv_layer = model.get_layer('conv2d_' + str(conv_count))
                conv_count+=1
                print("layer: ",i+1,conv_layer)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]
                if "batch_normalize" in block:
                    norm_layer = model.get_layer('bnorm_' + str(norm_count))
                    norm_count+=1
                    print("layer: ",i+1,norm_layer)
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                    # from the weight file the data is in format [beta ,gamma, mean, variance]
                    # tf batch_norm has [beta, gamma, mean, variance]
                    # hence need to do the [0 1 2 3] --> [1 0 2 3]

                    # all attributes of wieghts are stacked for each kernel in rows
                    # thus weights = [ [gamm1,  gamma2, gamma3, ....gamma_n ],
                    #                  [beta1,  beta2,  beta3,  ....beta_n  ],
                    #                  [mean1,  mean2,  mean3,  ....mean_n  ],
                    #                  [var1,   var2,   var3    ....var_n   ]]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                else:
                    conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
                # darknet shape (out_dim, in_dim, height, width)
                # tf shape (height, width, in_dim, out_dim)
                # hence need the transformation from [0 1 2 3] --> [2 3 1 0]
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
                if "batch_normalize" in block:
                    norm_layer.set_weights(bn_weights)
                    conv_layer.set_weights([conv_weights])
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])
        except ValueError:
            print('no conv_' + str(i+1))
    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()

def main(Args):
    model_size = (Args.model_size, Args.model_size, 3)
    YOLO_v3_Model = YOLOv3Net(Args.cfgfile, Args.num_classes, model_size, Args.max_total_size,\
                              Args.max_output_size_per_class, Args.iou_threshold, Args.score_threshold)
    load_weights(YOLO_v3_Model, Args.cfgfile, Args.weightfile)
    try:
        YOLO_v3_Model.save_weights('weights/yolo_v3_weights.tf')
        print('\nThe file \'yolo_v3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolo_v3_weights.tf\'.")
    return

if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--cfgfile', default= './cfg/yolov3.cfg', help='config file path, Default: ./cfg/yolov3.cfg')
    Parser.add_argument('--weightfile', default= './data/yolov3.weights', help='pre-trained weight file path, Default: ./data/yolov3.weights')
    Parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou_threshold for non-maximum suppression')
    Parser.add_argument('--score_threshold', type=float, default=0.5, help='score_threshold for non-maximum suppression')
    Parser.add_argument('--max_total_size', type=int, default=100, help='max_total_size for non-maximum suppression')
    Parser.add_argument('--max_output_size_per_class', type=int, default=100, help='max_output_size_per_class for non-maximum suppression')
    Parser.add_argument('--model_size', type=int, default=416, help='Input layer image size for the YOLO_V3 network. Input image will be resized to this size as a square image')
    Parser.add_argument('--num_classes', type=int, default=80, help='number of different objects that can be detected.')

    Args = Parser.parse_args()
    main(Args)