#!/bin/bash

PIPCMD=pip3

# 1. Install packages
# To solve this error -- import cv2 ImportError: libGL.so.1: cannot open shared object file
#sudo apt update 
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
  echo ${ID}
  sudo apt -y install libgl1-mesa-glx
  if [ $? -ne 0 ]; then
    apt -y install libgl1-mesa-glx
  fi
elif [[ ${ID} == "centos" ]]; then
  echo ${ID}
  sudo yum -y install mesa-libGL
  if [ $? -ne 0 ]; then
    yum -y install mesa-libGL
  fi
else
  echo "Unable to determine OS..."
fi

$PIPCMD install opencv-python
$PIPCMD install easydict
$PIPCMD install tqdm

# 2. Download datasets
if [ ! -d "VOC" ]; then
  wget -q /files/datasets/VOC/VOC.tar.gz
  tar xfz VOC.tar.gz
  rm -rf VOC.tar.gz
fi

# 3. Download pretrained yolov3 models
RUN_MODE=${RUN_MODE:-inference}
if [[ ${RUN_MODE} == "inference" ]]
then
  echo "Called inference"
  if [ ! -d "model_inference" ]; then
    wget -q /files/model/YOLOV3/model_inference.tar.gz
    mkdir model_inference
    tar xfz model_inference.tar.gz -C model_inference/
    rm -rf model_inference.tar.gz
  fi
elif [[ ${RUN_MODE} == "training" ]]
then
  echo "Called training"
  if [ ! -f "checkpoint/yolov3_coco_demo.ckpt.data-00000-of-00001" ]; then
    wget -q /files/model/YOLOV3/yolov3_coco_demo.ckpt.tar.gz
    tar xfz yolov3_coco_demo.ckpt.tar.gz -C checkpoint/
    rm -rf yolov3_coco_demo.ckpt.tar.gz
  fi
fi
