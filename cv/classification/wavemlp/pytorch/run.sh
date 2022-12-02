# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
DATA_PATH=$1

python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 train.py \
    ${DATA_PATH} \
    --output ./work_dir  \
    --model WaveMLP_T_dw \
    --sched cosine \
    --epochs 300 \
    --opt adamw \
    -j 8 \
    --warmup-lr 1e-6 \
    --mixup .8 \
    --cutmix 1.0 \
    --model-ema \
    --model-ema-decay 0.99996 \
    --aa rand-m9-mstd0.5-inc1 \
    --color-jitter 0.4 \
    --warmup-epochs 5 \
    --opt-eps 1e-8 \
    --repeated-aug \
    --remode pixel \
    --reprob 0.25  \
    --amp \
    --lr 1e-3 \
    --weight-decay .05 \
    --drop 0 \
    --drop-path 0.1 \
    -b 128 \
    --log-interval 1
