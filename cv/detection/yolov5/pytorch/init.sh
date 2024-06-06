#!/bin/bash
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

if [[ "$(uname)" == "Linux" ]]; then
    if command -v apt &> /dev/null; then
        apt install -y numactl libgl1-mesa-dev
    elif command -v yum &> /dev/null; then
        yum install -y numactl mesa-libGL
    else
        echo "Unsupported package manager"
        exit 1
    fi
else
    echo "Unsupported operating system"
    exit 1
fi

pip3 install -r requirements.txt
PY_VERSION=$(python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}')
if [ "$PY_VERSION" == "10" ]; then
   pip3 install matplotlib==3.8.2
   pip3 install numpy==1.22.4
   pip3 install Pillow==9.5 
else
   echo "only for python3.10"
fi

wandb disabled
pip3 install pycocotools
