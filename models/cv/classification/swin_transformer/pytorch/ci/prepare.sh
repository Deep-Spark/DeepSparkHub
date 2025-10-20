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

git clone https://github.com/microsoft/Swin-Transformer.git
git checkout f82860bfb5225915aca09c3227159ee9e1df874d
cd Swin-Transformer
pip install timm==0.4.12 yacs
## fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main.py

timeout 1800 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path ../imagenet --batch-size 128