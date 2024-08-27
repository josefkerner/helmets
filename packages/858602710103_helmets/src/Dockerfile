FROM public.ecr.aws/v3x9p5m3/davesmn-panorama-pytorch:latest

# Allow statements and log messages entities immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code entities the container image.
ENV APP_HOME /panorama

WORKDIR $APP_HOME

RUN python3.8 -m pip install --no-cache-dir boto3
COPY src /panorama