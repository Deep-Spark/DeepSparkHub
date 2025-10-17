#!/usr/bin/env bash

: ${IX_NUM_CUDA_VISIBLE_DEVICES:=2}
: ${DATA_DIR:="imagenette"}

: ${BATCH_SIZE:=32}
LOG_DIR="logs/resnet50"
BASE_DIR="./out_model"
MODEL_DIR=${BASE_DIR}/resnet50
WORK_PATH=$(dirname $(readlink -f $0))
OFFICALPATH=$WORK_PATH/../../../

export PYTHONPATH=$OFFICALPATH:$PYTHONPATH


EXIT_STATUS=0
check_status()
{
  if ((${PIPESTATUS[0]} != 0)); then
    EXIT_STATUS=1
  fi
}

if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
fi

if [ ! -d "${BASE_DIR}" ]; then
    mkdir -p ${BASE_DIR}
fi

rm -rf ${MODEL_DIR}


python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=${MODEL_DIR} \
  --data_dir=${DATA_DIR} \
  --config_file=configs/examples/resnet/imagenet/gpu_mirrored.yaml \
  --params_override="train_dataset.batch_size=${BATCH_SIZE}" \
  --params_override="runtime.num_gpus=${IX_NUM_CUDA_VISIBLE_DEVICES}$*";  check_status

exit ${EXIT_STATUS}
