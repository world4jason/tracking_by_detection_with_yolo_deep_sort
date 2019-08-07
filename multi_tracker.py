import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import detection as dt
from deep_sort.application_util import preprocessing


def splitInteger(m, n):
    assert n > 0
    quotient = m // n
    remainder = m % n
    if remainder > 0:
        return [0] + [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [0] + [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [0] + [quotient] * n


class Multi_tracker(object):
    def __init__(self,
                 tracker_type="",
                 max_cosine_distance=0.7,
                 nn_budget=100,
                 deep_sort_model_path=None,
                 labels=None,
                 nms_threshold=0.7):

        self.labels = labels
        self.nms_threshold = nms_threshold
        self.tracker_type = tracker_type
        self.nn_budget = nn_budget
        self.max_cosine_distance = max_cosine_distance
        self.trackers = [[] for i in range(len(labels))]
        self.encoder = None
        if deep_sort_model_path is not None:
            from deep_sort.tools import generate_detections as gen_dt
            self.encoder = gen_dt.create_box_encoder(deep_sort_model_path, batch_size=1)

    def gen_feature(self, image, boxes, confidences, classids, idxs, feature_maps, split):
        crop_features = [[] for i in range(len(self.labels))]

        level_3 = feature_maps[2][0]  # [256x(X/8)x(Y/8)]
        x_boundary = level_3.shape[2]
        y_boundary = level_3.shape[1]
        x_pool_ratio = (image.shape[1] / x_boundary) * 8
        y_pool_ratio = (image.shape[0] / y_boundary) * 8

        g_classids = [[] for i in range(len(self.labels))]
        g_confidences = [[] for i in range(len(self.labels))]
        g_boxes = [[] for i in range(len(self.labels))]
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                # min and max constrant for boundary
                # similiar to ROI pooling in Faster R-CNN
                # TODO: use interpolation
                x1 = max(0, min(x_boundary, round(boxes[i][0] // x_pool_ratio)))
                y1 = max(0, min(y_boundary, round(boxes[i][1] // y_pool_ratio)))
                x2 = max(0, min(x_boundary, round(boxes[i][2] // x_pool_ratio) + x1))
                y2 = max(0, min(y_boundary, round(boxes[i][3] // y_pool_ratio) + y1))

                # equal hanle
                x2 = x2 + 1 if x1 == x2 else x2
                y2 = y2 + 1 if y1 == y2 else y2

                # skip if lower than split size
                if (abs(y2 - y1) // split) == 0 and self.tracker_type in ["local_max_pooling","local_avg_pooling"]:
                    continue

                g_classids[classids[i]].append(classids[i])
                g_confidences[classids[i]].append(confidences[i])
                g_boxes[classids[i]].append(boxes[i])

                crop_feature = level_3[:, x1:x2, y1:y2]

                # global max pooling
                # np.max on x then y
                if self.tracker_type == "global_max_pooling":
                    gmp = np.max(crop_feature, axis=1)
                    crop_features[classids[i]].append(np.max(gmp, axis=1))

                # global average pooling
                # np.mean on x then y
                elif self.tracker_type == "global_avg_pooling":
                    gap = np.mean(crop_feature, axis=1)
                    crop_features[classids[i]].append(np.mean(gap, axis=1))

                # local max pooling
                elif self.tracker_type == "local_max_pooling":
                    split_lmp = []
                    split_range = splitInteger(abs(y2 - y1), split)  # split height into size of self.split
                    gmp = np.max(crop_feature, axis=1)
                    for j in range(1, split + 1):
                        split_range[j] += split_range[j - 1]
                    for j in range(split):
                        split_lmp.append(np.max(gmp[:, split_range[j]:split_range[j + 1]], axis=1))
                    crop_features[classids[i]].append(np.concatenate(split_lmp))

                # local average pooling
                if self.tracker_type == "local_avg_pooling":
                    split_lap = []
                    split_range = splitInteger(abs(y2 - y1), split)  # split height into size of self.split
                    gap = np.mean(crop_feature, axis=1)
                    for j in range(1, split + 1):
                        split_range[j] += split_range[j - 1]
                    for j in range(split):
                        split_lap.append(np.max(gap[:, split_range[j]:split_range[j + 1]], axis=1))
                    crop_features[classids[i]].append(np.concatenate(split_lap))

        # deep sort
        if self.tracker_type == "deep_sort":
            for index in range(len(self.labels)):
                if len(g_boxes[index]) == 0:
                    continue
                crop_features[index] = self.encoder(image, np.array(g_boxes[index]).copy())

        return g_boxes, g_confidences, g_classids, crop_features


    def predict(self, boxes, confidences, classids, crop_features):
        show_box = []
        show_confidence = []
        show_id = []
        show_name = []
        indices = []
        for index in range(len(self.labels)):
            if type(self.trackers[index]) is list:
                if len(classids[index]) != 0:
                    self.trackers[index] = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget))
                else:
                    continue
            detections = [
                dt.Detection(bbox, score, features, class_id)
                for bbox, score, features, class_id in
                zip(np.array(boxes[index]), np.array(confidences[index]), crop_features[index], classids[index])
            ]

            self.trackers[index].predict()
            self.trackers[index].update(detections)

            for track in self.trackers[index].tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                show_box.append(track.to_tlwh())
                show_confidence.append(track.confidence)
                show_id.append(track.track_id)
                show_name.append(track.get_name())

            indices = preprocessing.non_max_suppression(
                np.array(show_box), self.nms_threshold, np.array(show_confidence))
        return [show_box[i] for i in indices], [show_id[i] for i in indices], [show_name[i] for i in indices]
