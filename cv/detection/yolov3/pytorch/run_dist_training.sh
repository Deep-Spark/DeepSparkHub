#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p ${LOG_DIR}
fi
DATE=`date +%Y%m%d%H%M%S`

source ./get_num_devices.sh

# Run finetuning
python3 -m torch.distributed.launch --nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES --use_env \
    ./pytorchyolo/train.py --pretrained_weights checkpoints/yolov3_voc_pretrain.pth \
    --second_stage_steps 200  "$@" 2>&1 | tee ${LOG_DIR}/training_${DATE}.log

if [[ ${PIPESTATUS[0]} != 0 ]]; then
  echo "ERROR: finetuning on VOC failed"
  exit 1
fi

exit 0
