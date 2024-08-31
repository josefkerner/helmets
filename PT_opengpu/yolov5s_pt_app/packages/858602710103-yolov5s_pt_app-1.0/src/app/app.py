import panoramasdk as p
import numpy as np
import logging, os, glob
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
import sys
import cv2, math

os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"
import site
site.addsitedir('/usr/lib/python3.8/site-packages/')
site.addsitedir('/usr/local/lib/python3.8/site-packages/')

classNames = [
    "helmet", "head", "person"
]


class Frame:
    def __init__(self, img):
        self.image = img



from ultralytics import YOLO
model_path = '/panorama/yolov5s_model/yolov5s_half.pt'
#model_path = 'helmets/packages/858602710103-helmets-2.0/src/app/yolov5s_half.pt'
model = YOLO(model_path)


class ObjectDetectionApp(p.node):

        def __init__(self):
            self.pre_processing_output_size = 640

        def get_frames(self):
            input_frames = self.inputs.video_in.get()
            return input_frames

        def process_media(self, frames):
            final_frames = []
            for frame in frames:
                results = model(frame.image)

                for r in results:
                    boxes = r.boxes

                    coords = []
                    objects = []

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # put box in cam
                        cv2.rectangle(frame.image, (x1, y1), (x2, y2), (255, 0, 255), 3)

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
                        print(objects)
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

                        cv2.putText(frame.image, txt, coords[0], font, fontScale, color, thickness)
                        final_frames.append(frame)
            return final_frames

        def mock_input_frames(self):


            outputs = []
            frame_rate = 30
            video_duration=1
            width = 640
            height = 480
            frames = []
            for _ in range(int(frame_rate * video_duration)):
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                obj = Frame(img=frame)
                outputs.append(obj)

            return outputs


        def run(self):
            print('app starts')
            log.info("Pytorch Yolov5s FP16 App starts")
            while True:
                input_frames = []
                try:
                    input_frames = self.get_frames()
                    print('got this frame number: ',len(input_frames))
                    if len(input_frames) == 0:
                        raise ValueError('no camera input')
                    print(input_frames[0].image)
                except Exception as e:
                    #input_frames = self.mock_input_frames()
                    print('NO CAMERA INPUT')
                    print(e)

                if len(input_frames) != 0:
                    output_frames = self.process_media(input_frames)
                    self.outputs.video_out.put(output_frames)

current_directory = os.getcwd()
print(current_directory)
try:
    app = ObjectDetectionApp()
    app.run()
except Exception as err:
    log.exception('App did not Start {}'.format(err))
    sys.exit(1)