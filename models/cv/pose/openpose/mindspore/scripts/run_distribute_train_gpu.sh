#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Copyright (c) 2023,Shanghai Iluvatar CoreX Semiconductor Co.,Ltd.
# All Rights Reserved.

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
if [ $# != 4 ]
then
    echo "Usage: sh scripts/run_distribute_train_gpu.sh [IAMGEPATH_TRAIN] [JSONPATH_TRAIN] [MASKPATH_TRAIN] [VGG_PATH]"
exit 1
fi

export DEVICE_NUM=8

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

WORKDIR=./train_parallel
rm -rf $WORKDIR
mkdir $WORKDIR
cp ./*.py $WORKDIR
cp -r ./src $WORKDIR
cp ./*yaml $WORKDIR
cp -r ./scripts $WORKDIR
cd $WORKDIR || exit
echo "start distributed training with $DEVICE_NUM GPUs."
env >env.log
mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
  python3 train.py \
    --imgpath_train=$1 \
    --jsonpath_train=$2 \
    --maskpath_train=$3 \
    --vgg_path=$4 > log.txt 2>&1 &
  cd ..
