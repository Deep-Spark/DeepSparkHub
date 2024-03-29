#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0
EXECUTE_PATH=$(pwd)
train_path=${EXECUTE_PATH}/s16_aug_train_1p

if [ -d ${train_path} ]; then
  rm -rf ${train_path}
fi
mkdir -p ${train_path}
mkdir ${train_path}/device${DEVICE_ID}
mkdir ${train_path}/ckpt
cd ${train_path}/device${DEVICE_ID} || exit

python3 ${EXECUTE_PATH}/../train.py --data_file=$1/vocaug_train.mindrecord0  \
                    --train_dir=${train_path}/ckpt  \
                    --device_target="GPU" \
                    --train_epochs=200  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --base_lr=0.015  \
                    --lr_type=cos  \
                    --min_scale=0.5  \
                    --max_scale=2.0  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --ckpt_pre_trained=$1/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt  \
                    --save_steps=1500  \
                    --keep_checkpoint_max=200 >log 2>&1 &
