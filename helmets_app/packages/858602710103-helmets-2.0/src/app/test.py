import panoramasdk as p

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
import sys
import cv2, math




from ultralytics import YOLO
model = YOLO("/panorama/yolov5s_half.pt")