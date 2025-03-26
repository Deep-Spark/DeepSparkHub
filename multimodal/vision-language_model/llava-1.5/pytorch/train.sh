#!/bin/bash

# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.                                                                                                                                                             
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

export ENABLE_FLASH_ATTENTION_WITH_IXDNN=0
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_CONVERT_ATTN_MASK_TO_INF=1

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/llava-1.5-7b-hf \
    --version plain \
    --data_path data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder data/LLaVA-Pretrain/images \
    --vision_tower checkpoints/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
#    --report_to wandb
