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
## install libGL
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y libgl1-mesa-glx
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL
else
    echo "Not Support Os"
fi

mkdir -p /home/data/perf/ssd
ln -s /mnt/deepspark/data/datasets/coco /home/data/perf/ssd/
cp /mnt/deepspark/data/checkpoints/resnet34-333f7ec4.pth /home/data/perf/ssd/

cd base
source ../iluvatar/config/environment_variables.sh
python3 prepare.py --name iluvatar --data_dir /home/data/perf/ssd
timeout 1800 bash run_training.sh --name iluvatar --config V100x1x8 --data_dir /home/data/perf/ssd --backbone_path /home/data/perf/ssd/resnet34-333f7ec4.pth