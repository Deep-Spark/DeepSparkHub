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

mkdir -p data/
ln -s /mnt/deepspark/data/datasets/coco2017 data/coco
mkdir -p /root/.cache/torch/hub/checkpoints/
cp /mnt/deepspark/data/checkpoints/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/

cd start_scripts
bash init_torch.sh

timeout 1800 bash train_fasterrcnn_resnet50_amp_torch.sh --dataset coco --data-path ../data/coco

# bash train_fasterrcnn_resnet50_amp_dist_torch.sh --dataset coco --data-path ../data/coco
