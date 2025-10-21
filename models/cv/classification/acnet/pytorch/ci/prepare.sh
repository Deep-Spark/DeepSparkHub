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

git clone https://github.com/DingXiaoH/ACNet.git
cd ACNet
ln -s /mnt/deepspark/data/datasets/imagenet ./imagenet_data
rm -rf acnet/acb.py
rm -rf utils/misc.py
mv ../acb.py acnet/
mv ../misc.py utils/
export PYTHONPATH=$PYTHONPATH:.
# export CUDA_VISIBLE_DEVICES=0
timeout 1800 python3 -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b acb --local_rank 0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# timeout 1800 python3 -m torch.distributed.launch --nproc_per_node=8 acnet/do_acnet.py -a sres18 -b acb