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
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
DATA_DIR=$1
CKPT_PATH=$2

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config_gpu.yaml"

mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
  python3 ./train.py --config_path=$CONFIG_FILE --is_distributed=True --platform 'GPU' \
  --dataset_path $DATA_DIR --ckpt_path=$CKPT_PATH > train.log 2>&1 
