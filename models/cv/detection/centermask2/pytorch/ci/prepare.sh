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

# Install detectron2 with 9604f5995cc628619f0e4fd913453b4d7d61db3f
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2

# 6adabd66034347b7da07f2d474e4baa1c27b54ee is the commit hash of centermask2
git clone https://github.com/youngwanLEE/centermask2.git
cd centermask2
# fix RuntimeError: "max_elementwise_cuda" not implemented for 'Double'
sed -i 's@torch.max(mask_ratios, value_eps)@torch.max(mask_ratios.cpu(), value_eps.cpu()).cuda()@' centermask/modeling/centermask/mask_head.py
sed -i 's@torch.max(mask_union_area, value_1)@torch.max(mask_union_area.cpu(), value_1.cpu()).cuda()@' centermask/modeling/centermask/mask_head.py
sed -i 's@torch.max(mask_ovr_area, value_0)@torch.max(mask_ovr_area.cpu(), value_0.cpu()).cuda()@' centermask/modeling/centermask/mask_head.py

mkdir -p datasets/
ln -s /mnt/deepspark/data/datasets/coco2017 datasets/coco
mkdir -p /root/.torch/iopath_cache/detectron2/ImageNetPretrained/MSRA/
cp /mnt/deepspark/data/checkpoints/R-50.pkl /root/.torch/iopath_cache/detectron2/ImageNetPretrained/MSRA/
timeout 1800 python3 train_net.py --config-file configs/centermask/centermask_R_50_FPN_ms_3x.yaml --num-gpus 4