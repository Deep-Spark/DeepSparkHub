#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

## install libGL
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y libgl1-mesa-glx
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL
else
    echo "Not Support Os"
fi

## clone yolov5 and install
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov5/pytorch/
pip3 install -r requirements.txt
pip3 install matplotlib==3.8.2
pip3 install numpy==1.26.4
pip3 install pandas==2.0.0
wandb disabled
pip3 install pycocotools

mkdir -p datasets
ln -s /mnt/deepspark/data/datasets/coco ./datasets/coco

### On single GPU
timeout 1800 python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights ''

### On single GPU (AMP)
# python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights '' --amp

### Multiple GPUs on one machine
# python3 -m torch.distributed.launch --nproc_per_node 8 \
#     train.py \
#     --data ./data/coco.yaml \
#     --batch-size 64 \
#     --cfg ./models/yolov5s.yaml --weights '' \
#     --device 0,1,2,3,4,5,6,7

# YOLOv5m
# bash run.sh

# ### Multiple GPUs on one machine (AMP)
# python3 -m torch.distributed.launch --nproc_per_node 8 \
#     train.py \
#     --data ./data/coco.yaml \
#     --batch-size 256 \
#     --cfg ./models/yolov5s.yaml --weights '' \
#     --device 0,1,2,3,4,5,6,7 --amp


## Test the detector
# python3 detect.py --source ./data/images/bus.jpg --weights yolov5s.pt --img 640
# python3 detect.py --source ./data/images/zidane.jpg --weights yolov5s.pt --img 640