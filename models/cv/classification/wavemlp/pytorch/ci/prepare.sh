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
pip install thop timm==0.4.5 torchprofile
git clone https://github.com/huawei-noah/Efficient-AI-Backbones.git
cd Efficient-AI-Backbones/wavemlp_pytorch/
git checkout 25531f7fdcf61e300b47c52ba80973d0af8bb011

# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' train.py
# change dataset
sed -i "s@from timm.data import Dataset@from timm.data import ImageDataset@" train.py
sed -i "s@dataset_train = Dataset(train_dir)@dataset_train = ImageDataset(train_dir)@" train.py
sed -i "s@dataset_eval = Dataset(eval_dir)@dataset_eval = ImageDataset(eval_dir)@" train.py
sed -i 's/args.max_history/100/g' train.py

python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 train.py ../imagenet/ --output ./output/  --model WaveMLP_T_dw --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 5 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path 0.1 -b 128