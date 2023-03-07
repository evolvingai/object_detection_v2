import queue

import cv2
import setproctitle as setproctitle
import time

DEBUG = False

def run(framesQueue, stream_rtsp, FLAGS):

    setproctitle.setproctitle("Decoding")

    try:
        capture = cv2.VideoCapture(stream_rtsp)
    except:

        print("Error geting stream")
        return


    while True:
        time1 = time.time()
        _, frame = capture.read()
        resized = cv2.resize(frame, (1920, 1080))
        #print("putting")
        try:
            framesQueue.put(resized)
        except queue.Full:
            pass

        if DEBUG:

            time2 = time.time()
            print("Decoding + Resize + Put", time2-time1)
            cv2.imshow("bla", frame)
            cv2.waitKey(1)



