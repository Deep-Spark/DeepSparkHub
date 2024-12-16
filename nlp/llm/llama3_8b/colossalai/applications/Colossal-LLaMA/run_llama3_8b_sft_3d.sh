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

PARENT_SAVE_DIR="checkpoint"
PARENT_TENSORBOARD_DIR="tensorboard"
PARENT_CONFIG_FILE="config"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG_DIR="logs/${TIMESTAMP}"
SAVE_DIR="${LOG_DIR}/${PARENT_SAVE_DIR}"
TENSORBOARD_DIR="${LOG_DIR}/${PARENT_TENSORBOARD_DIR}"
CONFIG_FILE="${LOG_DIR}/${PARENT_CONFIG_FILE}.json"

DATASET_PATH=./dataset/school_math/convert/llama3_data_sft/arrow/
TOKENIZER_DIR=/home/model_zoos/nlp/Meta-Llama-3-8B
GLOBAL_BATCH_SIZE_PER_DP=8
MICRO_BATCH_SIZE=1


mkdir -p $LOG_DIR
colossalai run --nproc_per_node 16 train.py \
    --config "llama3_8b" \
    --dataset  $DATASET_PATH \
    --tokenizer_dir $TOKENIZER_DIR \
    --max_length 8192 \
    --plugin "3d" \
    --zero_stage 1 \
    --pp 4 \
    --custom_ckpt \
    --custom_recompute_layers_per_stage 7 6 5 6 \
    --ignore_steps 2 \
    --save_interval 0 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 1 \
    --batch_size $GLOBAL_BATCH_SIZE_PER_DP \
    --microbatch_size $MICRO_BATCH_SIZE \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --use_flash_attn \
    --use_neft \
    --pad_token "eos" |& tee ${LOG_DIR}/output.log

