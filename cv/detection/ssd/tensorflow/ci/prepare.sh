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

pip3 install tf_slim
cd dataset
mkdir tfrecords
ln -s /mnt/deepspark/data/datasets/VOC2012_sample ./
python3 convert_voc_sample_tfrecords.py --dataset_directory=./ --output_directory=tfrecords --train_splits VOC2012_sample --validation_splits VOC2012_sample
cd ..
ln -s /mnt/deepspark/data/checkpoints/ssd-vgg16 ./model
python3 train_ssd.py --batch_size 16