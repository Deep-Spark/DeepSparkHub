#!/bin/bash

bash ./get_imagenette.sh

export TF_CUDNN_USE_AUTOTUNE=1
export TF_CPP_MIN_LOG_LEVEL=1

: ${BATCH_SIZE:=32}
#TRAIN_EPOCHS=10
# optional optimizer: momentum, rmsprop, momentum, sgd
OPTIMIZER=momentum
DATE=`date +%Y%m%d%H%M%S`

LOG_DIR="logs/alexnet"
DATA_DIR=./imagenette
BASE_DIR=train_dir
TRAIN_DIR=${BASE_DIR}/alexnet

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

python3 -u tf_cnn_benchmarks.py\
 --data_name=imagenette --data_dir=${DATA_DIR}\
 --data_format=NCHW --batch_size=${BATCH_SIZE}\
 --model=alexnet --optimizer=${OPTIMIZER} --num_gpus=1\
 --weight_decay=1e-4 --train_dir=${TRAIN_DIR}\
 --eval_during_training_every_n_epochs=2\
 --num_eval_epochs=1 --datasets_use_caching\
 --stop_at_top_1_accuracy=0.9\
 --num_intra_threads=1 --num_inter_threads=1 "${new_args[@]}" 2>&1  | tee ${LOG_DIR}/${DATE}_${TRAIN_EPOCHS}_${BATCH_SIZE}_${OPTIMIZER}.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit


exit ${EXIT_STATUS}
