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

export CLIP_FLASH_ATTN=1
export USE_NHWC_GN=1
export USE_IXFORMER_GEGLU=0
export USE_APEX_LN=1
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1


export MODEL_PATH=${MODEL_PATH:-runwayml/stable-diffusion-v1-5}
export DATASET_PATH=${DATASET_PATH:-lambdalabs/pokemon-blip-captions}

accelerate launch --config_file configs/single_config.yaml --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --dataset_name=$DATASET_PATH \
  --resolution=512 \
  --seed 42 \
  --center_crop \
  --random_flip \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-1.5-single" \
  --max_train_steps=15000 \
  --NHWC \
  --dataloader_num_workers=32 \
  --apex_fused_adam 
  # --use_ema
