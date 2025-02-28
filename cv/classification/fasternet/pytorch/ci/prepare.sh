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

pip install -r requirements.txt
git clone https://github.com/JierunChen/FasterNet.git
cd FasterNet
git checkout e8fba4465ae912359c9f661a72b14e39347e4954
timeout 1800 python3 train_test.py -g 0,1,2,3 --num_nodes 1 -n 4 -b 4096 -e 2000 \
                      --data_dir ../imagenet \
                      --pin_memory --wandb_project_name fasternet \
                      --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') \
                      --cfg cfg/fasternet_t0.yaml