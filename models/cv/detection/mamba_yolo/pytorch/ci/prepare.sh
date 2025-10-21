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

pip3 install seaborn thop timm einops

mkdir -p /root/.config/Ultralytics/
cp /mnt/deepspark/data/3rd_party/Arial.ttf /root/.config/Ultralytics/

git clone --depth 1 https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/mamba-yolo/pytorch

cd selective_scan && pip install . && cd ..
pip install -v -e .

mkdir -p /root/data
ln -s /mnt/deepspark/data/datasets/coco /root/data/coco
sed -i 's/\/mnt\/datasets\/MSCOCO2017/\/root\/data\/coco/g' ultralytics/cfg/datasets/coco.yaml
ln -s /mnt/deepspark/data/checkpoints/yolov8n.pt ./

timeout 1800 python3 mbyolo_train.py --task train --data ultralytics/cfg/datasets/coco.yaml \
 --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
--amp  --project ./output_dir/mscoco --name mambayolo_n --device 0,1,2,3