# Introdction
This is a multiple object tracking project that use Yolov3(opencv ver.) as detector and [DeepSort](https://github.com/nwojke/deep_sort) as trackers.
<div>
Different from original method in Deepsort. I did some improvement.

1. Remove the network and reuse the feature maps from Yolov3 with following methods:

    1.1 : global average pooling<div>

    1.2 : global max pooling<div>

    1.3 : Part-based average pooling<div>

    1.4 : Part-based max pooling<div>


2. Extend the method from only tracking poeple to all the objects that the detector detected.

Videos demo on Cityscapes (record at 1 fps)
1. [Video 1](https://youtu.be/MF_788uv5uQ)
2. [Video 2](https://youtu.be/w7_eK_M1ycQ)
3. [Video 3](https://youtu.be/5JZVdT43fAc)

# Installation
1. Clone the repository:
````
git clone https://github.com/world4jason/tracking_by_detection
````
2. Download pre-generated detections and the CNN checkpoint file from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) and the netowrk folder under model_data

3. Download YOLOv3 weights from [Yolo Website](http://pjreddie.com/darknet/yolo/) and put into model_data/yolov3 folder or runs

````
sh tools/get_yolo.sh
````



# Dependencies
See requirements.txt

Additionally, network feature generation for original Deepsort method requires TensorFlow-1.4.0. or TensorFlow-1.5.0.

# Running

````
python3 demo.py
````

### Parameters
````
usage: demo.py [-h] [--conf_threshold CONF_THRESHOLD]
               [--nms_threshold NMS_THRESHOLD] [--net_width NET_WIDTH]
               [--net_height NET_HEIGHT] [--tracker_type TRACKER_TYPE]
               [--split SPLIT] [--tracker_nms_threshold TRACKER_NMS_THRESHOLD]
               [--max_cosine_distance MAX_COSINE_DISTANCE]
               [--nn_budget NN_BUDGET] [--label_path LABEL_PATH]
               [--model_path MODEL_PATH] [--weight_path WEIGHT_PATH]
               [--sort_model_path SORT_MODEL_PATH]
               [--output_video OUTPUT_VIDEO] [--video_path VIDEO_PATH]

YoloV3 with Variants Sorts

optional arguments:
  -h, --help            show this help message and exit
  --conf_threshold CONF_THRESHOLD
  --nms_threshold NMS_THRESHOLD
  --net_width NET_WIDTH
  --net_height NET_HEIGHT
  --tracker_type TRACKER_TYPE
  --split SPLIT
  --tracker_nms_threshold TRACKER_NMS_THRESHOLD
  --max_cosine_distance MAX_COSINE_DISTANCE
  --nn_budget NN_BUDGET
  --label_path LABEL_PATH
  --model_path MODEL_PATH
  --weight_path WEIGHT_PATH
  --sort_model_path SORT_MODEL_PATH
  --output_video OUTPUT_VIDEO
  --video_path VIDEO_PATH
````
# License
[world4jason/tracking_by_detection](https://github.com/world4jason/tracking_by_detection) is released under the GPL-3.0.
