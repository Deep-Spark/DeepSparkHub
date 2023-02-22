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

if [ $# != 3 ]
then 
    echo "Usage: sh run_eval_gpu.sh [DATASET_PATH] [DATASET_NAME] [CKPT]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$1
DATASET_NAME=$2
echo $DATASET_NAME
MODEL_CKPT=$(get_real_path $3)
echo $MODEL_CKPT


if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp *.sh ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cd ./eval || exit
echo "start eval on standalone GPU"

if [ $DATASET_NAME == cora ]
then
    python eval.py --data_dir=$DATASET_PATH --device_target="GPU"  --model_ckpt $MODEL_CKPT &> log &
fi

if [ $DATASET_NAME == citeseer ]
then
    python eval.py --data_dir=$DATASET_PATH --device_target="GPU"  --model_ckpt $MODEL_CKPT &> log &
fi
cd ..
