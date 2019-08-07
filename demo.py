import numpy as np
import cv2

from yolov3 import Yolov3
from multi_tracker import Multi_tracker


def main(args):
    # label setting
    labels = open(args.label_path).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # video setting
    fps = 1.0
    camera = cv2.VideoCapture(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (video_width, video_height))

    # detector setting
    detector = Yolov3(
        model_path=args.model_path,
        weight_path=args.weight_path,
        net_width=video_width,  # args.net_width,
        net_height=video_height,  # args.net_height,
        conf_threshold=0.7,
        nms_threshold=0.7)

    # tracker setting
    track_method_list = [
        "global_max_pooling", "global_avg_pooling", "deep_sort",
        "local_max_pooling", "local_avg_pooling",
    ]
    assert args.tracker_type in track_method_list
    mot = Multi_tracker(
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
        tracker_type=args.tracker_type,
        nms_threshold=args.tracker_nms_threshold,
        labels=labels
    )



    while True:
        (grabbed, image) = camera.read()
        if not grabbed:
            break
        boxes, confidences, classids, idxs, feature_maps = detector.detect(image)
        boxes, confidences, classids, crop_features = mot.gen_feature(  image,
                                                                        boxes,
                                                                        confidences,
                                                                        classids,
                                                                        idxs,
                                                                        feature_maps,
                                                                        args.split)

        bboxes, track_ids, names = mot.predict(boxes, confidences, classids, crop_features)

        for idx in range(len(bboxes)):
            # draw
            color = [int(c) for c in COLORS[names[idx]]]
            cv2.rectangle(image,
                        (int(bboxes[idx][0]), int(bboxes[idx][1])),
                        (int(bboxes[idx][0] + bboxes[idx][2]), int(bboxes[idx][1] + bboxes[idx][3])), color, 2  )
            text = "{}".format(track_ids[idx])
            cv2.putText(
                image, text,
                (int(bboxes[idx][0]), int(bboxes[idx][1] - 5  )),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4  , color,  1  )

        cv2.imshow('ouputs', image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Cleanup
    camera.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    from os.path import dirname, abspath

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YoloV3 with Variants Sorts')

    parser.add_argument('--conf_threshold', required=False, default=0.5)
    parser.add_argument('--nms_threshold', required=False, default=0.4)
    parser.add_argument('--net_width', required=False, default=416)
    parser.add_argument('--net_height', required=False, default=416)
    parser.add_argument('--tracker_type', required=False, default="global_max_pooling")
    parser.add_argument('--split', required=False, default=4)
    parser.add_argument('--tracker_nms_threshold', required=False, default=0.7)
    parser.add_argument('--max_cosine_distance', required=False, default=0.2)
    parser.add_argument('--nn_budget', required=False, default=100)
    parser.add_argument('--label_path', required=False,
        default=dirname(abspath(__file__)) + "/model_data/coco.names")
    parser.add_argument('--model_path', required=False,
        default=dirname(abspath(__file__)) + "/model_data/yolov3/yolov3.cfg")
    parser.add_argument('--weight_path', required=False,
        default=dirname(abspath(__file__)) + "/model_data/yolov3/yolov3.weights")
    parser.add_argument(
        '--sort_model_path',
        required=False,
        default=dirname(abspath(__file__)) +
        "/model_data/networks/mars-small128.pb")
    parser.add_argument(
        '--output_video', required=False, default='demo.avi')
    parser.add_argument(
        '--video_path', required=False, default="AVG-TownCentre.mp4")

    main(parser.parse_args())
