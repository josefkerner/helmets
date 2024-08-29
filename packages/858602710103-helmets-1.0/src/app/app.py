import panoramasdk as p

import logging, os, glob
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
import sys
import cv2, math

classNames = [
    "helmet", "head", "person"
]




from ultralytics import YOLO
#model_path = '/panorama/app/best.pt'
model_path = 'helmets/packages/858602710103-helmets-1.0/src/app/best.pt'
model = YOLO(model_path)

class ObjectDetectionApp(p.node):

        def __init__(self):
            self.pre_processing_output_size = 640

        def get_frames(self):
            input_frames = self.inputs.video_in.get()
            return input_frames

        def process_media(self, image_list):
            final_image_list = []
            for img in image_list:
                results = model(img, stream=True)

                for r in results:
                    boxes = r.boxes

                    coords = []
                    objects = []

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # put box in cam
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        print("Confidence --->", confidence)

                        # class name
                        cls = int(box.cls[0])
                        if cls > len(classNames):
                            obj = "Unknown"
                            print("Class name -->", "Unknown")
                        else:
                            obj = model.names[cls]
                            objects.append(obj)
                            # print("Class name -->", classNames[cls])

                        # object details
                        org = [x1, y1]
                        coords.append([x1, y1])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    if len(coords) > 0:
                        if len(coords) > 1:
                            distance = math.sqrt(
                                (coords[0][0] - coords[1][0]) ** 2 + (coords[0][1] - coords[1][1]) ** 2)
                        else:
                            distance = 'no'

                        if len(objects) == 1 and objects[0] == 'helmet':
                            txt = f"Wears helmet"
                        else:
                            txt = f"No helmet"

                        cv2.putText(img, txt, coords[0], font, fontScale, color, thickness)
                        final_image_list.append(img)
            return final_image_list

        def run(self):
            print('app starts')
            log.info("Pytorch Yolov5s FP16 App starts")
            image_list = [] # An image queue
            while True:
                input_frames = self.get_frames()
                image_list += [frame.image for frame in input_frames]
                image_list = self.process_media(image_list)
                self.outputs.video_out.put(image_list)

current_directory = os.getcwd()
print(current_directory)
try:
    app = ObjectDetectionApp()
    app.run()
except Exception as err:
    log.exception('App did not Start {}'.format(err))
    sys.exit(1)