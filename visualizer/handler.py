import multiprocessing

import setproctitle as setproctitle
import cv2
from utils.utils import *
model_input_shape = [640, 640]
orig_shape = [1920, 1080]
import time
import yaml
CONF_TH = 0.3

def run(predictionsQueue, FLAGS):
    setproctitle.setproctitle("Visualizer")
    frameIndex = 0
    color = (255, 0, 255)
    thickness = 2
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    with open("utils/classes.yaml", "r") as stream:
        try:
            data_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    while True:


        try:

            data = predictionsQueue.get()
            if not data:
                continue
            time1 = time.time()
            predictions1 = data['c1']
            predictions2 = data['c2']
            names = data_yaml['names']

            decoded_frame = data['frame']

            frame = decoded_frame.copy()
            frame2 = decoded_frame.copy()

            for det in predictions1:
                det[:4] = scale_boxes(model_input_shape, det[:4], frame.shape).round()
                xyxy = det[:4]
                conf = det[4]
                cls = det[5]
                if conf < CONF_TH:
                    continue

                frame = cv2.rectangle(frame, (int(xyxy[0] - 10), int(xyxy[1])), (int(xyxy[2] - 10), int(xyxy[3])),
                                      color=(255, 0, 0),
                                      thickness=3)
                frame = cv2.putText(
                    img=frame,
                    text=names[int(cls)],
                    org=(int(xyxy[0]), int(xyxy[3])),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=3
                )
            for det2 in predictions2:
                det2[:4] = scale_boxes(model_input_shape, det2[:4], frame.shape).round()
                xyxy = det2[:4]
                conf = det2[4]
                if conf < CONF_TH:
                    continue
                cls = det2[5]

                frame2 = cv2.rectangle(frame2, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                       color=colors(int(cls), True),
                                       thickness=3)
                frame2 = cv2.putText(
                    img=frame2,
                    text=names[int(cls)],
                    org=(int(xyxy[0]), int(xyxy[1])),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=colors(int(cls), True),
                    thickness=3
                )
            # Generate output by blending image with shapes image, using the shapes
            # images also as mask to limit the blending to those parts

            cv2.imshow('Visualizer Predictions M1 ', cv2.resize(frame, (640, 640)))
            cv2.imshow('Visualizer Predictions M2', cv2.resize(frame2, (640, 640)))


            cv2.waitKey(1)
            frameIndex += 1
            time2 = time.time()

            print("Time Show", time2 - time1)

        except:
            pass




