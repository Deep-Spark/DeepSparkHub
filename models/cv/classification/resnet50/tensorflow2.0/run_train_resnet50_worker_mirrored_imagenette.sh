#!/usr/bin/env bash

LOG_DIR="logs/resnet50"
DATA_DIR="imagenette"
BASE_DIR="./out_model"
MODEL_DIR=${BASE_DIR}/resnet50
DATE=`date +%Y%m%d%H%M%S`
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



for index in 0 1
do
  export CUDA_VISIBLE_DEVICES=${index}
  time python3 classifier_trainer.py \
    --mode=train_and_eval \
    --model_type=resnet \
    --dataset=imagenet \
    --model_dir=${MODEL_DIR} \
    --data_dir=${DATA_DIR} \
    --config_file=configs/examples/resnet/imagenet/gpu_multi_worker_mirrored.yaml \
    --params_override='runtime.num_gpus=2, runtime.task_index='${index}''  2>&1 | tee ${LOG_DIR}/${DATE}_${index}.log [[ ${PIPESTATUS[0]} == 0 ]] || exit &
done

wait
if [ ! -f "compare_kv.py" -o ! -f "get_key_value.py" ]; then
  ./download_script.sh
  if [[ $? != 0 ]]; then
    echo "ERROR: download scripts failed"
    exit 1
  fi
fi
echo ${DATE}
python3 get_key_value.py -i ${LOG_DIR}/${DATE}_0.log -k 'val_accuracy: ' 'val_top_5_accuracy: ' -o train_resnet50_worker_mirrored_bi.json
python3 compare_kv.py -b train_resnet50_worker_mirrored_bi.json -n train_resnet50_worker_mirrored_nv.json -i 'val_accuracy: ' 'val_top_5_accuracy: '; check_status
exit ${EXIT_STATUS}
