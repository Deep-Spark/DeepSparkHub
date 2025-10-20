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

pip3 install timm yacs
git clone https://github.com/DingXiaoH/RepVGG.git
cd RepVGG
git checkout eae7c5204001eaf195bbe2ee72fb6a37855cce33
# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main.py

# change dataset load
# Tips: "import os" into data/build.py
sed -i "s@dataset = torchvision.datasets.ImageNet(root=config.DATA.DATA_PATH, split='train' if is_train else 'val', transform=transform)@dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, prefix), transform=transform)@" data/build.py

timeout 1800 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12349 main.py --arch RepVGG-A0 --data-path ../imagenet --batch-size 32 --tag train_from_scratch --output ./ --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 1e-4 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET weak AUG.MIXUP 0.0 DATA.DATASET imagenet DATA.IMG_SIZE 224