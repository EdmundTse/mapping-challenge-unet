FROM tensorflow/tensorflow:2.4.1-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt install -y cmake git protobuf-compiler

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Install TensorFlow 2 Object Detection models
RUN git clone --depth 1 https://github.com/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install --user --use-feature=2020-resolver .

ENV TF_CPP_MIN_LOG_LEVEL 3

# Install Nvidia and project dependencies
RUN pip install  --user git+https://github.com/NVIDIA/dllogger

ADD requirements.txt /home/tensorflow/requirements.txt
RUN python -m pip install -r /home/tensorflow/requirements.txt  --user --use-feature=2020-resolver

# Switch to the directory where the project will be mounted
WORKDIR /code
