import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
from botocore.exceptions import CredentialRetrievalError
import cv2
import math
import numpy as np
import panoramasdk
from typing import List, Dict
from ultralytics import YOLO
model_path = '/panorama/best.pt'
#model_path = 'helmets/packages/858602710103-helmets-2.0/src/app/yolov5s_half.pt'
model = YOLO(model_path)

classNames = [
    "helmet", "head", "person"
]

class Application(panoramasdk.node):

    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.MODEL_NODE = 'model_node'
        self.MODEL_DIM = 224
        self.frame_num = 0
        self.detected_frame = 0
        self.inference_time_ms = 0
        self.inference_time_max = 0
        self.frame_time_ms = 0
        self.frame_time_max = 0
        self.epoch_frames = 150
        self.epoch_start = time.time()
        self.detected_class = None
        logger.info('## ENVIRONMENT VARIABLES\r{}'.format(dict(**os.environ)))
        try:
            # Parameters
            logger.info('Configuring parameters.')
            self.threshold = self.inputs.threshold.get()
            self.device_id = self.inputs.device_id.get()
            self.log_level = self.inputs.log_level.get()
            self.region = self.inputs.region.get()
            # logger
            if self.log_level in ('DEBUG','INFO','WARNING','ERROR','CRITICAL'):
                logger.setLevel(self.log_level)
            # read classes
            with open('/panorama/squeezenet_classes.json','r') as f:
                self.classes= json.load(f)
        except:
            logger.exception('Error during initialization.')
        try:
            # AWS SDK
            logger.info('Configuring AWS SDK for Python.')
            boto_session = boto3.session.Session(region_name=self.region)
            self.cloudwatch = boto_session.resource('cloudwatch')
        except CredentialRetrievalError:
            logger.warn('AWS SDK credentials are not available. Disabling metrics.')
        except:
            logger.exception('Error creating AWS SDK session.')
        finally:
            logger.info('Initialization complete.')

    def postprocess_labels(self,img, detections: List[Dict]):
        for detection in detections:
            print(detection)

            coords = [detection['x1'], detection['y1']]
            if detection['label'] == "helmet":
                # count heads
                heads = [d for d in detections if d['label'] == 'head']
                # if confidence is greater than 0.5
                if detection['confidence'] > 0.5:
                    # if there are no heads
                    txt = f'Helmet ON - {detection["confidence"]}'
                else:
                    txt = 'No helmet'
            else:
                txt = 'No helmet'
                x1 = detection['x1']
                y1 = detection['y1']
                y2 = detection['y2']
                x2 = detection['x2']
            # if text starts with 'Wearing helmet'
            if txt.startswith('Wearing helmet'):
                # set color to green
                color = (0, 255, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2

                cv2.putText(img, txt, coords, font, fontScale, color, thickness)

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        frame_start = time.time()
        self.frame_num += 1
        logger.debug(self.frame_num)
        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream in streams:
            if stream.is_cached:
                return
            self.process_media(stream)
        # Log metrics
        frame_time = (time.time() - frame_start) * 1000
        if frame_time > self.frame_time_max:
            self.frame_time_max = frame_time
        self.frame_time_ms += frame_time
        if self.frame_num % self.epoch_frames == 0:
            epoch_time = time.time() - self.epoch_start
            epoch_fps = self.epoch_frames/epoch_time
            avg_inference_time = self.inference_time_ms / self.epoch_frames / len(streams)
            max_inference_time = self.inference_time_max
            avg_frame_processing_time = self.frame_time_ms / self.epoch_frames
            max_frame_processing_time = self.frame_time_max
            logger.info('epoch length: {:.3f} s ({:.3f} FPS)'.format(epoch_time, epoch_fps))
            logger.info('avg inference time: {:.3f} ms'.format(avg_inference_time))
            logger.info('max inference time: {:.3f} ms'.format(max_inference_time))
            logger.info('avg frame processing time: {:.3f} ms'.format(avg_frame_processing_time))
            logger.info('max frame processing time: {:.3f} ms'.format(max_frame_processing_time))
            self.inference_time_ms = 0
            self.inference_time_max = 0
            self.frame_time_ms = 0
            self.frame_time_max = 0
            self.epoch_start = time.time()
            self.put_metric_data('AverageInferenceTime', avg_inference_time)
            self.put_metric_data('AverageFrameProcessingTime', avg_frame_processing_time)

        self.outputs.video_out.put(streams)

    def process_media(self, stream):
        """Runs inference on a frame of video."""
        #image_data = preprocess(stream.image,self.MODEL_DIM)
        #logger.debug('Image data: {}'.format(image_data))
        # Run inference
        inference_start = time.time()
        #inference_results = self.call({"data":image_data}, self.MODEL_NODE)
         # Log metrics
        inference_time = (time.time() - inference_start) * 1000
        if inference_time > self.inference_time_max:
            self.inference_time_max = inference_time
        self.inference_time_ms += inference_time
        # Process results (classification)
        self.process_results(stream.image)

    def process_results(self, stream):
        '''
        :param stream: is single frame
        :return:
        '''

        results = model(stream, stream=True)
        objects = []

        for r in results:
            boxes = r.boxes


            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(stream, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                if cls > len(classNames):
                    label = "Unknown"
                    print("Class name -->", "Unknown")
                else:
                    label = model.names[cls]

                    obj = {
                        "label": label,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                    objects.append(obj)
                    # print("Class name -->", classNames[cls])

                # object details

        self.postprocess_labels(stream, objects)

    def put_metric_data(self, metric_name, metric_value):
        """Sends a performance metric to CloudWatch."""
        namespace = 'AWSPanoramaApplication'
        dimension_name = 'Application Name'
        dimension_value = 'aws-panorama-sample'
        try:
            metric = self.cloudwatch.Metric(namespace, metric_name)
            metric.put_data(
                Namespace=namespace,
                MetricData=[{
                    'MetricName': metric_name,
                    'Value': metric_value,
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {
                            'Name': dimension_name,
                            'Value': dimension_value
                        },
                        {
                            'Name': 'Device ID',
                            'Value': self.device_id
                        }
                    ]
                }]
            )
            logger.info("Put data for metric %s.%s", namespace, metric_name)
        except ClientError:
            logger.warning("Couldn't put data for metric %s.%s", namespace, metric_name)
        except AttributeError:
            logger.warning("CloudWatch client is not available.")

def preprocess(img, width, color_order_out='RGB', color_order_in='BGR'):
    """Resizes and normalizes a frame of video."""
    resized = cv2.resize(img, (width, width))
    channels = {
        'R': {
            'mean': 0.485,
            'std' : 0.229
        },
        'G': {
            'mean': 0.456,
            'std' : 0.224
        },
        'B': {
            'mean': 0.406,
            'std' : 0.225
        }
    }
    img = resized.astype(np.float32) / 255.
    img_r = img[:, :, color_order_in.index('R')]
    img_g = img[:, :, color_order_in.index('G')]
    img_b = img[:, :, color_order_in.index('B')]
    # normalize each channel and flatten
    x1 = [[[], [], []]]
    x1[0][color_order_out.index('R')] = (img_r - channels['R']['mean']) / channels['R']['std']
    x1[0][color_order_out.index('G')] = (img_g - channels['G']['mean']) / channels['G']['std']
    x1[0][color_order_out.index('B')] = (img_b - channels['B']['mean']) / channels['B']['std']
    return np.asarray(x1)

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    LOG_PATH = '/opt/aws/panorama/logs'
    handler = RotatingFileHandler("{}/app.log".format(LOG_PATH), maxBytes=10000000, backupCount=1)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    try:
        logger.info("INITIALIZING APPLICATION")
        logger.info('Numpy version %s' % np.__version__)
        app = Application()
        logger.info("PROCESSING STREAMS")
        while True:
            app.process_streams()
            # turn off debug logging after 150 loops
            if logger.getEffectiveLevel() == logging.DEBUG and app.frame_num == 150:
                logger.setLevel(logging.INFO)
    except:
        logger.exception('Exception during processing loop.')

logger = get_logger(level=logging.INFO)
main()