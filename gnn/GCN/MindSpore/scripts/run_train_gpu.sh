#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 2 ]
then 
    echo "Usage: sh run_train_gpu.sh [DATASET_PATH] [DATASET_NAME]"
exit 1
fi

DATASET_PATH=$1
DATASET_NAME=$2
echo $DATASET_NAME

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit
env > env.log
echo "start training for standalone GPU"


if [ $DATASET_NAME == cora ]
then
    python3 train.py --data_dir=$DATASET_PATH --train_nodes_num=140 --device_target="GPU" &> log &
fi

if [ $DATASET_NAME == citeseer ]
then
    python3 train.py --data_dir=$DATASET_PATH --train_nodes_num=120 --device_target="GPU" &> log &
fi
cd ..
