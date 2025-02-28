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

pip3 install -r requirements.txt
git clone https://github.com/xingyizhou/CenterNet.git
cd CenterNet
git checkout 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c
# Compile deformable convolutional(DCNv2)
cd ./src/lib/models/networks/
rm -rf DCNv2
git clone -b pytorch_1.11 https://github.com/lbin/DCNv2.git
cd ./DCNv2/
python3 setup.py build develop
cd ../../../../../
ln -s /mnt/deepspark/data/datasets/coco2017 ./data/coco
mkdir -p /root/.cache/torch/hub/checkpoints/
cp /mnt/deepspark/data/checkpoints/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/

cd ./src
touch lib/datasets/__init__.py
timeout 1800 python3 main.py ctdet --arch res_18 --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0