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

#bash ./get_imagenet_1000.sh

export TF_CUDNN_USE_AUTOTUNE=1
export TF_CPP_MIN_LOG_LEVEL=1

: ${BATCH_SIZE:=32}
#TRAIN_EPOCHS=10
# optional optimizer: momentum, rmsprop, momentum, sgd
OPTIMIZER=rmsprop
DATE=`date +%Y%m%d%H%M%S`

LOG_DIR="logs/inceptionV3_multigpu"
DATA_DIR=./imagenet_tfrecord
BASE_DIR=train_dir
TRAIN_DIR=${BASE_DIR}/inceptionV3_multigpu

mkdir -p ${LOG_DIR}
mkdir -p ${BASE_DIR}
rm -rf ${TRAIN_DIR}

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

i=0
for arg in "$@"
do
    if [[ $arg =~ "--epoch" ]]; then
        new_args[$i]="--num_epochs"
    else
        new_args[$i]=$arg
    fi
    let i++
done

source ./get_num_devices.sh

UMD_WAITAFTERLAUNCH=1 python3 -u tf_cnn_benchmarks.py\
 --data_name=imagenet --data_dir=${DATA_DIR}\
 --data_format=NCHW --batch_size=${BATCH_SIZE}\
 --model=inception3 --optimizer=${OPTIMIZER} --num_gpus=${IX_NUM_CUDA_VISIBLE_DEVICES}\
 --weight_decay=1e-4 --train_dir=${TRAIN_DIR}\
 --eval_during_training_every_n_epochs=2\
 --num_eval_epochs=1 --datasets_use_caching\
 --stop_at_top_1_accuracy=0.9 --all_reduce_spec=pscpu\
 --num_intra_threads=1 --num_inter_threads=1 "${new_args[@]}" 2>&1 | tee ${LOG_DIR}/${DATE}_${TRAIN_EPOCHS}_${BATCH_SIZE}_${OPTIMIZER}.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit


exit ${EXIT_STATUS}
