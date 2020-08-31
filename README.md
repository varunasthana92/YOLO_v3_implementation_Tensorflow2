# YOLO_V3 object detection implementation in tensorflow2 using pre-trained model

## Overview
["YOLOv3: An Incremental Improvement" ](https://arxiv.org/pdf/1804.02767.pdf) paper can be accessed from [here](https://arxiv.org/pdf/1804.02767.pdf). 

<div style='text-align: justify'>
YOLO is used for multiple object detection in a colored image. Version-3 supports detection of 80 different objects. The original model was trained on COCO dataset (for more details refer the paper). YOLO_V3 model is generated using the architecture "yolov3.cfg" config file provided by the authors. Complete end-to-end network architecture flowchart is provided in the "Model.png" file. For easy understanding of the config file block numbers have been added in the config file and the custom config file is provided in the "cfg" directory. Block numbers are not to be confused with the layer numbers. Any reference to a layer number are "0" based index numbers, like in "route" or "shortcut" blocks in the config file.
</div>

<p align="center">
<img src="https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2/blob/master/Result/street.jpg">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2/blob/master/Result/freeway.jpg">
</p>

## Dependencies
* python 3.5.2
* OpenCV 4.1.2
* numpy 1.18.1
* tensorflow 2.2.0

## How to run

Download the pre-trained weights of the YOLO_V3 provided by the authors of the original paper from [here](https://pjreddie.com/media/files/yolov3.weights) and save it in the 'data/' sub-directory. Now run the below command to convert the weights to tensorflow compatible format.

```
git clone https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2.git
mkdir weights
<download the pre-trained weight file>
python3 convert_weights.py
```

Above commands are to be executed only once. The converted weights can then be used with the tensorflow implementation using the below command.
```
python3 detect.py --img_path=data/images/street.jpg
```

__Implementation Notes:__
* Anchor box sizes (provided in the cfg file) are to be normalized with the model size (input layer image size). 

## Contact Information
Name: Varun Asthana  
Email id: varunasthana92@gmail.com

## References
* https://arxiv.org/pdf/1804.02767.pdf
* https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6
* https://github.com/zzh8829/yolov3-tf2
