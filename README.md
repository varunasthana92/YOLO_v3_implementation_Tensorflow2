# YOLO_V3 object detection implementation in tensorflow2 using pre-trained model

### Overview
[Link](https://arxiv.org/pdf/1804.02767.pdf) to the paper "YOLOv3: An Incremental Improvement" 

YOLO is used for multiple object detection in a colored image. Version_3 supports detection of 80 different objects. The original model was trained on COCO dataset (for more details refer the paper).

<p align="center">
<img src="https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2/blob/master/Result/street.jpg">
</p>





```
git clone https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2.git
```
### Dependencies
* python 3.5.2
* OpenCv 4.1.2
* numpy 1.18.1
* tensorflow 2.2.0  
The program has been tested to work for the versions shows above

### Run Command
Download the pre-trained weights of the YOLO_V3 provided by the authors of the original paper from [here](https://pjreddie.com/media/files/yolov3.weights) and save it in the 'data/' sub-directory. Now run the below command to convert the weights to tensorflow compatible format.

```
python3 convert_weights.py
```


```
python3 detect.py --img_path=data/images/street.jpg
```

