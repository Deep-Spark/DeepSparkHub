#!/bin/bash
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
