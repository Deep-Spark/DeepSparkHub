#!/bin/bash

# Sentence paraphrase classification tasks
# Dataset: GLUE/MRPC
# Model: bert-base

MODEL="bert-base_uncased_L-12_H-768_A-12"
DATE=`date +%Y%m%d%H%M%S`
WORK_PATH=$(dirname $(readlink -f $0))
OFFICALPATH=$WORK_PATH/../../../
LOG_DIR="logs/bert"
BERT_DIR=pretrained_model/uncased_L-12_H-768_A-12
MODEL_DIR=output_dir_
GLUE_DIR=./datasets
TASK=MRPC
BATCH=32
GPUS=${1:-'0,1'}
WORKER_HOSTS=${2:-'localhost:20001','localhost:20002'}
# GPUS=${1:-'0,1,2,3,4,5,6,7'}
# WORKER_HOSTS=${2:-'localhost:20001','localhost:20002','localhost:20003','localhost:20004','localhost:20005','localhost:20006','localhost:20007','localhost:20008'}

export PYTHONPATH=$OFFICALPATH:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_VLOG_LEVEL=0

mkdir -p ${LOG_DIR}
rm -rf ${MODEL_DIR}*

EXIT_STATUS=0
check_status()
{
  if ((${PIPESTATUS[0]} != 0)); then
    EXIT_STATUS=1
  fi
}

# Download model
if [ ! -d ${BERT_DIR} ]; then
  mkdir -p ${BERT_DIR}
  cp ${WORK_PATH}/../../../../data/model_zoo/bert/${MODEL}/* ${BERT_DIR}
fi

# Download data
if [ ! -d ${GLUE_DIR}/${TASK} ]; then
  mkdir -p ${GLUE_DIR}/${TASK}
  cp ${WORK_PATH}/../../../../data/datasets/MRPC_tf_record/* ${GLUE_DIR}/${TASK}
fi

pip3 install tensorflow_hub tensorflow_addons gin-config

gpus_array=(`echo $GPUS | tr ',' ' '` ) 

for ((index=0; index<${#gpus_array[@]}; index++))
do
  export CUDA_VISIBLE_DEVICES=${gpus_array[index]}
  time python3 run_classifier.py \
    --mode='train_and_eval' \
    --input_meta_data_path=${GLUE_DIR}/${TASK}/${TASK}_meta_data \
    --train_data_path=${GLUE_DIR}/${TASK}/${TASK}_train.tf_record \
    --eval_data_path=${GLUE_DIR}/${TASK}/${TASK}_eval.tf_record \
    --bert_config_file=${BERT_DIR}/bert_config.json \
    --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
    --train_batch_size=${BATCH} \
    --eval_batch_size=${BATCH} \
    --steps_per_loop=1 \
    --learning_rate=2e-5 \
    --num_train_epochs=3 \
    --model_dir=${MODEL_DIR}${index} \
    --distribution_strategy=multi_worker_mirrored \
    --task_index=${index} \
    --worker_hosts=${WORKER_HOSTS} 2>&1 | tee ${LOG_DIR}/${BATCH}_${DATE}_${index}.log || exit &
done

wait
if [ ! -f "compare_kv.py" -o ! -f "get_key_value.py" ]; then
  ./download_script.sh
  if [[ $? != 0 ]]; then
    echo "ERROR: download scripts failed"
    exit 1
  fi
fi

python3 get_key_value.py -i ${LOG_DIR}/${BATCH}_${DATE}_0.log -k 'loss: ' 'accuracy: ' 'val_loss: ' 'val_accuracy: ' -o train_worker_mirrored_bi.json
python3 compare_kv.py -b train_worker_mirrored_bi.json -n train_worker_mirrored_nv.json -i 'val_accuracy: '; check_status
exit ${EXIT_STATUS}