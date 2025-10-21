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
git clone https://github.com/THU-MIG/RepViT.git
cd RepViT
git checkout 298f42075eda5d2e6102559fad260c970769d34e
pip3 install -r requirements.txt
# On single GPU
timeout 1800 python3 main.py --model repvit_m0_9 --data-path ../imagenet --dist-eval

# # Multiple GPUs on one machine
# python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py --model repvit_m0_9 --data-path ./imagenet --dist-eval