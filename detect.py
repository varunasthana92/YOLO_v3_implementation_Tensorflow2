#! /usr/bin/env python
import tensorflow as tf
from utils import resize_image, load_class_names, draw_outputs, SetGPU
import cv2
import numpy as np
from model import YOLOv3Net
import argparse

def main(Args):
    model_size = (Args.model_size, Args.model_size, 3)
    model = YOLOv3Net(Args.cfgfile, Args.num_classes, model_size, Args.max_total_size,\
                      Args.max_output_size_per_class, Args.iou_threshold, Args.score_threshold)
    model.load_weights(Args.weightfile)
    # tf.keras.utils.plot_model(model, to_file='Model.png', show_shapes=True)
    class_names = load_class_names(Args.class_name)
    image = cv2.imread(Args.img_path)
    original_image_size = image.shape
    org_img = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.expand_dims(image, 0)
    resized_frame = resize_image(image, (model_size[0], model_size[1]))
    boxes, scores, classes, num_detections = model.predict(resized_frame)
    print('Number of detections = ', num_detections)
    img_final = draw_outputs(org_img, boxes, scores, classes, num_detections, class_names)
    cv2.imwrite('./Result/' + Args.img_path.split('/')[-1], img_final)
    cv2.imshow('Detections', img_final)
    print('Press ESC on image display wondow to exit')
    cv2.waitKey(0)


if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--class_name', default= './data/coco.names', help='File path for class names, Default: ./data/coco.names')
    Parser.add_argument('--cfgfile', default= './cfg/yolov3.cfg', help='config file path, Default: ./cfg/yolov3.cfg')
    Parser.add_argument('--weightfile', default= './weights/yolo_v3_weights.tf', help='tensorflow weight file path, Default: ./weights/yolo_v3_weights.tf')
    Parser.add_argument('--img_path', default= './data/images/freeway.jpg', help='image path for detection, Default: .data/images/freeway.jpg')
    Parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou_threshold for non-maximum suppression')
    Parser.add_argument('--score_threshold', type=float, default=0.5, help='score_threshold for non-maximum suppression')
    Parser.add_argument('--max_total_size', type=int, default=100, help='max_total_size for non-maximum suppression')
    Parser.add_argument('--max_output_size_per_class', type=int, default=100, help='max_output_size_per_class for non-maximum suppression')
    Parser.add_argument('--model_size', type=int, default=416, help='Input layer image size for the YOLO_V3 network. Input image will be resized to this size as a square image')
    Parser.add_argument('--num_classes', type=int, default=80, help='number of different objects that can be detected.')
    Parser.add_argument('--GPUDevice', type=int, default= -1, help='What GPU do you want to use? -1 for CPU, Default:-1')    
    
    Args = Parser.parse_args()
    SetGPU(Args.GPUDevice)
    main(Args)