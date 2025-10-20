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
git clone https://github.com/DingXiaoH/RepMLP.git
cd RepMLP
git checkout 3eff13fa0257af28663880d870f327d665f0a8e2
# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main_repmlp.py
# change dataset load
sed -i "s@dataset = torchvision.datasets.ImageNet(root=config.DATA.DATA_PATH, split='train' if is_train else 'val', transform=transform)@dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, prefix), transform=transform)@" data/build.py

timeout 1800 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12349 main_repmlp.py --arch RepMLPNet-B256 --batch-size 32 --tag my_experiment --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.001 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 10 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256 --data-path ../imagenet