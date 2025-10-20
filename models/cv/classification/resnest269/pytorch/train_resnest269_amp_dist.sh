#!/bin/bash -ex
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
# Modify me
if [ -z "$IMAGENETTE" ];then
	DATA_PATH=${DATA_PATH:-"/home/datasets/cv/imagenet"}
else
	# for quick test
	DATA_PATH="/home/datasets/cv/imagenette"
fi

# Don't modify
CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}
PYTHONARG='-m torch.distributed.launch --nproc_per_node auto --use_env --master_port 10016'
# For debugging
if [ ! -z "${DEBUG}" ];then
	PYTHONAR="${PYTHONAR} -m pdb"
fi
cd ${ROOT_DIR}
python3  $PYTHONARG ${ROOT_DIR}/run_train.py \
	--model resnest269 --dali --dali-cpu  --data-path $DATA_PATH \
	--opt fused_sgd --batch-size 32 --crop-size 416 \
	--lr 0.001 --amp --nhwc --deterministic "$@"
