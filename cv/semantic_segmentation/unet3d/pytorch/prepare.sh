#!/bin/bash
# Copyright (c) 2022-2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
apt update -y
apt install -y numactl git wget

# echo "install packages..."


#pip3 install "git+https://github.com/mlperf/logging.git@0.7.0"
#pip3 install -r requirements.txt
#python3 setup.py install

echo "prepare data..."

mkdir -p /home/datasets/cv/kits19/train

data_file="/home/datasets/cv/kits19/train/kits19.tar.gz"

if [ ! -e ${data_file} ]; then
    echo "ERROR: Invalid data file ${data_file}"
    # wget -P /home/datasets/cv/kits19/train /url/to/kits19.tar.gz
fi

echo "Uncompressing the kits19!"
tar -xf ${data_file} -C /home/datasets/cv/kits19/train


