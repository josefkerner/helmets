FROM public.ecr.aws/panorama/panorama-application:1.2.0-py3.8


WORKDIR /panorama
COPY . .

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install --no-cache-dir torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install --no-cache-dir opencv-python