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

if [ $# != 4 ]
then
    echo "Usage: bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$1
CHECKPOINT_PATH=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $4)
echo $DATASET
echo $CONFIG_PATH
echo $CHECKPOINT_PATH

if [ ! -f $CHECKPOINT_PATH ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$3
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit

if [ -d "eval$3" ];
then
    rm -rf ./eval$3
fi

mkdir ./eval$3
cp ./*.py ./eval$3
cp ./config/*.yaml ./eval$3
cp -r ./src ./eval$3
cd ./eval$3 || exit
env > env.log
echo "start inferring for device $DEVICE_ID"
python eval.py \
    --dataset=$DATASET \
    --checkpoint_file_path=$CHECKPOINT_PATH \
    --device_id=$3 \
    --config_path=$CONFIG_PATH > log.txt 2>&1 &
cd ..
