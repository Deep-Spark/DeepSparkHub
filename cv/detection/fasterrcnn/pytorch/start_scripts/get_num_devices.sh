#!/bin/bash
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
devices=$CUDA_VISIBLE_DEVICES
if [ -n "$devices"  ]; then
    _devices=(${devices//,/ })
    num_devices=${#_devices[@]}
else
    num_devices=2
    export CUDA_VISIBLE_DEVICES=0,1
    echo "Not found CUDA_VISIBLE_DEVICES, set nproc_per_node = ${num_devices}"
fi
export IX_NUM_CUDA_VISIBLE_DEVICES=${num_devices}
