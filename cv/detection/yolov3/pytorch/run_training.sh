#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p ${LOG_DIR}
fi
DATE=`date +%Y%m%d%H%M%S`

# Run finetuning
python3 pytorchyolo/train.py "$@" 2>&1 | tee ${LOG_DIR}/training_${DATE}.log

if [[ ${PIPESTATUS[0]} != 0 ]]; then
  echo "ERROR: finetuning on VOC failed"
  exit 1
fi

exit 0
