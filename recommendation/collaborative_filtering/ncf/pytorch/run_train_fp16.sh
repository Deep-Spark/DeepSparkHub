#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

python3 -m torch.distributed.launch --nproc_per_node=8 \
                      --master_port 10002 \
 ncf_16.py data/ml-20m \
  -l  "0.03" \
  -e   5      \
  -b  "1048576" \
  -b1 "0.25" \
  -b2 "0.5" \
  --eps "1e-8" \
  --valid-batch-size "1048576" \
  --loss-scale 8192  \
  --layers  256 256 128 64 -f 64  \
  --seed 1234 \
  --threshold 0.635 \
  --fp16 2>&1 |tee ncf_fp16.txt