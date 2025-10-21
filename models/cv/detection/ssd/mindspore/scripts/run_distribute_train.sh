#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh DEVICE_NUM EPOCH_SIZE LR DATASET RANK_TABLE_FILE  CONFIG_PATH PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_distribute_train.sh 8 500 0.2 coco /data/hccl.json /config_path /opt/ssd-300.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 6 ] && [ $# != 8 ]
then
    echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] \
[RANK_TABLE_FILE] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $6)
# Before start distribute train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True --dataset=$4 --config_path=$CONFIG_PATH

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$1
EPOCH_SIZE=$2
LR=$3
DATASET=$4
PRE_TRAINED=$7
PRE_TRAINED_EPOCH_SIZE=$8
export RANK_TABLE_FILE=$5

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    rm -rf LOG$i
    mkdir ./LOG$i
    cp ./*.py ./LOG$i
    cp ./config/*.yaml ./LOG$i
    cp -r ./src ./LOG$i
    cd ./LOG$i || exit
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    if [ $# == 6 ]
    then
        python train.py  \
        --run_distribute=True  \
        --lr=$LR \
        --dataset=$DATASET \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --epoch_size=$EPOCH_SIZE \
        --config_path=$CONFIG_PATH \
        --output_path './output' > log.txt 2>&1 &
    fi

    if [ $# == 8 ]
    then
        python train.py  \
        --run_distribute=True  \
        --lr=$LR \
        --dataset=$DATASET \
        --device_num=$RANK_SIZE  \
        --device_id=$DEVICE_ID  \
        --pre_trained=$PRE_TRAINED \
        --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
        --epoch_size=$EPOCH_SIZE \
        --config_path=$CONFIG_PATH \
        --output_path './output' > log.txt 2>&1 &
    fi

    cd ../
done
