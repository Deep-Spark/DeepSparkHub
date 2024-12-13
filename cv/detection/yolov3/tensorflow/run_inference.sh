#!/bin/bash
# Run yolov3 inference on Pascal VOC dataset, and evaluate the results.

RUN_MODE="inference" bash ./setup.sh
python3 evaluate.py
if [[ $? != 0 ]]; then
  echo "ERROR: run tf-yolov3 evalute.py failed!"
  exit 1
fi
cd mAP
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p ${LOG_DIR}
fi
DATE=`date +%Y%m%d%H%M%S`
python3 main.py -na -np 2>&1 | tee ${LOG_DIR}/inference_${DATE}.log
if [[ $? != 0 ]]; then
  echo "ERROR: get tf-yolov3 mAP stats failed!"
  exit 1
fi

exit 0
