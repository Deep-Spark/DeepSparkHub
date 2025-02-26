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
export USE_IXFORMER_GEGLU=1
export USE_APEX_LN=1
export USE_NATIVE_ATTN=0
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1


export MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3-medium-diffusers}
export DATASET_PATH=${DATASET_PATH:-diffusers/dog-example}
export OUTPUT_DIR="trained-sd3"

accelerate launch --main_process_port 29501 --config_file configs/zero2_config.yaml --mixed_precision="fp16" train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --instance_data_dir=$DATASET_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_epochs=25 \
  --seed="0" \
  --NHWC \
  --apex_fused_adam
