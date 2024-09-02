#!/bin/bash
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

NUM_GPUS=16
CONFIG_FILE=$1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# Check if a second argument is provided for NUM_GPUS
if [ ! -z "$2" ]; then
  NUM_GPUS=$2
fi

echo "start training with num_gpus=$NUM_GPUS | and config_file=$CONFIG_FILE"

# modeling_chatglm.py 有一点修改，用原生的会导致sft训练报错
cp -r models/* checkpoint/chatglm3-6b

torchrun  --nnodes=1 --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS  finetune_hf.py \
    data/AdvertiseGen_process/  \
    checkpoint/chatglm3-6b \
    $CONFIG_FILE
