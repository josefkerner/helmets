FROM public.ecr.aws/v3x9p5m3/davesmn-panorama-pytorch:latest

RUN python -m pip install --upgrade pip

WORKDIR /panorama
COPY src /panorama

#RUN install panorama-sdk
RUN python3 -m pip install panorama-sdk

#install opencv
RUN python3 -m pip install opencv-python-headless
#install ultralytics yolov8
RUN python3 -m pip install ultralytics