import panoramasdk as p

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
import sys
import cv2, math




from ultralytics import YOLO
model = YOLO("/panorama/best.pt")

class ObjectDetectionApp(p.node):

        def __init__(self):
            self.pre_processing_output_size = 640

        def get_frames(self):
            input_frames = self.inputs.video_in.get()
            return input_frames

        def run(self):
            log.info("Pytorch Yolov5s FP16 App starts")
            image_list = [] # An image queue
        def get_frames(self):
            input_frames = self.inputs.video_in.get()
            return input_frames

if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception('App did not Start {}'.format(err))
        sys.exit(1)