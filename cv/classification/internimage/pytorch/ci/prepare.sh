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

## Install mmcv
cd mmcv/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
cd ../

## Install timm and mmdet
pip3 install timm==0.6.11 mmdet==2.28.1

pip3 install addict yapf opencv-python termcolor yacs pyyaml scipy

cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
# python3 test.py
cd ../
# Training on 8 GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LOCAL_SIZE=8
# python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/internimage_t_1k_224.yaml --data-path ./imagenet

# Training on 1 GPU
export CUDA_VISIBLE_DEVICES=0
export LOCAL_SIZE=1
timeout 1800 python3 main.py --cfg configs/internimage_t_1k_224.yaml --data-path ./imagenet