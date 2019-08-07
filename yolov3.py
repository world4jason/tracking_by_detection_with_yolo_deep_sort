import numpy as np
import cv2


class Yolov3(object):
    def __init__(self,
                 conf_threshold=0.5,
                 nms_threshold=0.4,
                 net_width=416,
                 net_height=416,
                 model_path="",
                 weight_path=""):
        # Initialize the parameters
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net_width = net_width
        self.net_height = net_height
        self.net = cv2.dnn.readNetFromDarknet(model_path, weight_path)
        self.layer_names = self.net.getLayerNames()
        self.detection_layer = [
            self.layer_names[i[0] - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]
        self.feautre_maps = [
            self.layer_names[i[0] - 6]
            for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, frame):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0, (self.net_width, self.net_height),
            swapRB=True,
            crop=False)
        self.net.setInput(blob)

        # ['yolo_82', 'conv_80', 'yolo_94', 'conv_92', 'yolo_106', 'conv_104']
        layerOutputs = self.net.forward(self.detection_layer + self.feautre_maps)

        detections = layerOutputs[:2]
        feature_maps = layerOutputs[3:]

        boxes, confidences, classids = self.post_process( detections, height, width, self.conf_threshold)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        return boxes, confidences, classids, idxs, feature_maps

    def post_process(self, outs, height, width, conf_threshold):
        boxes = []
        confidences = []
        classids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classid = np.argmax(scores)
                confidence = scores[classid]

                if confidence > conf_threshold:
                    box = detection[0:4] * np.array(
                        [width, height, width, height])
                    centerX, centerY, bwidth, bheight = box.astype('int')

                    # Using the center x, y coordinates to derive the top
                    # and the left corner of the bounding box
                    x = int(centerX - (bwidth / 2))
                    y = int(centerY - (bheight / 2))

                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

        return boxes, confidences, classids
