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

# Run yolov3 training on Pascal VOC with pretrained model

# RUN_MODE="training" bash ./setup.sh
TRAIN_MODE=${TRAIN_MODE:-fast}
if [[ ${TRAIN_MODE} == "fast" ]]
then
  # Fast training: 2 epochs for first stage, 100 steps for second stage
  python3 train_fast.py "$@"
elif [[ ${TRAIN_MODE} == "slow" ]]
then
  # Slow training: 4 epochs for first stage, 4 epochs for second stage
  python3 train.py "$@"
else
  echo "TRAIN_MODE: Wrong value! Only accept fast or slow"
  exit $ERRCODE
fi

if [[ $? != 0 ]]; then
  echo "ERROR: run tf-yolov3 training failed!"
  exit 1
fi
# Evaluate model and calculate mAP
python3 evaluate_fast.py "$@"
if [[ $? != 0 ]]; then
  echo "ERROR: run tf-yolov3 evaluate_fast.py failed!"
  exit 1
fi
cd mAP
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p ${LOG_DIR}
fi
DATE=`date +%Y%m%d%H%M%S`
python3 main.py -na -np
if [[ $? != 0 ]]; then
  echo "ERROR: get tf-yolov3 mAP stats failed!"
  exit 1
fi

exit 0
