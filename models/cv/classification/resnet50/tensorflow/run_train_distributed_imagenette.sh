#!/bin/bash

bash ./get_imagenette.sh

export TF_CUDNN_USE_AUTOTUNE=1
export TF_CPP_MIN_LOG_LEVEL=1

#################################################
# Prepare training arguments
#################################################

i=0
model="alexnet"
for arg in "$@"
do
    if [ $i -eq 0 ]; then
        model=$arg
        let i++
        continue
    fi
    if [[ $arg =~ "--epoch" ]]; then
        new_args[$i]="--num_epochs"
    else
        new_args[$i]=$arg
    fi
    let i++
done
echo "## Training model: ${model}"


: ${BATCH_SIZE:=160}
TRAIN_EPOCHS=30
# optional optimizer: momentum, rmsprop, momentum, sgd
OPTIMIZER=momentum
DATE=`date +%Y%m%d%H%M%S`

LOG_DIR="logs/${model}_distributed"
DATA_DIR=./imagenette
BASE_DIR=train_dir
TRAIN_DIR=${BASE_DIR}/${model}_distributed

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

#################################################
# Prepare devices
#################################################
devices=$CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
if [ -n "$devices"  ]; then
    devices=(${devices//,/ })
    num_devices=${#devices[@]}

    if [ "${num_devices}" -gt "2" ]; then
        devices=(${devices[@]:0:2})
        num_devices=2

        VALID_DEVICES="${devices[0]},${devices[1]}"
        export CUDA_VISIBLE_DEVICES=${VALID_DEVICES}
        echo "Distributed_replicated mode, use 2 gpus train: ${VALID_DEVICES}"
    fi
else
    devices=(0 1)
    VALID_DEVICES="0,1"
    export CUDA_VISIBLE_DEVICES=${VALID_DEVICES}
    num_devices=2
fi
echo "devices: ${devices}"
echo "num_devices: ${num_devices}"

if [ "${num_devices}" == "1" ]; then
    echo "Error: The number of devices must be greater then 1 for distributed training, but got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
    exit 0
fi

#################################################
# Prepare distributed training arguments
#################################################
worker_hosts=""
i=0
for device in "${devices[@]}";
do
    if [ "$i" == "0" ]; then
        let i++
        continue
    fi
    let i++
    port=$((53400 + i))
    worker_hosts="${worker_hosts},127.0.0.1:${port}"
done
worker_hosts=${worker_hosts#*,}
echo "worker_hosts: ${worker_hosts}"

#################################################
# Handle CTRL-C
#################################################
trap ctrl_c INT
function ctrl_c() {
  echo "*** Trapped CTRL-C, killing process running background"
  for pid in "${pid_list[@]}"; do
    echo "Killing pid ${pid}"
    kill ${pid}
    wait ${pid}
  done
  exit 0
}

#################################################
# Start distributed training
#################################################

pid_list=()
last_device=`expr ${num_devices} - 1`
i=0
for device in "${devices[@]}";
do
    job_name="worker"
    if [ "${i}" == "0" ]; then
        job_name="ps"
    fi

    if [ ${i} -le 1 ]; then
        task_index=0
    else
        task_index=`expr ${i} - 1`
    fi

    if [ "${i}" == "${last_device}" ]; then
        echo "device: ${device}"
        UMD_WAITAFTERLAUNCH=1 python3 -u tf_cnn_benchmarks.py\
         --data_name=imagenette --data_dir=${DATA_DIR}\
         --data_format=NCHW \
         --optimizer=${OPTIMIZER} --datasets_use_prefetch=False\
         --local_parameter_device=gpu --num_gpus=${num_devices}\
         --batch_size=${BATCH_SIZE} --model=${model} \
         --variable_update=distributed_replicated \
         --num_epochs=${TRAIN_EPOCHS} \
         --job_name=${job_name} --ps_hosts=127.0.0.1:53400 --worker_hosts="${worker_hosts}"\
         --train_dir=${TRAIN_DIR} --task_index=${task_index} --print_training_accuracy=True "${new_args[@]}" 2>&1 | tee ${LOG_DIR}/${DATE}_${TRAIN_EPOCHS}_${BATCH_SIZE}_${OPTIMIZER}.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit
        echo "Distributed training on device ${device} where job name = ${job_name}"
    else
        echo "device: ${device}"
        UMD_WAITAFTERLAUNCH=1 python3 -u tf_cnn_benchmarks.py\
         --data_name=imagenette --data_dir=${DATA_DIR}\
         --data_format=NCHW \
         --optimizer=${OPTIMIZER} --datasets_use_prefetch=False\
         --local_parameter_device=gpu --num_gpus=${num_devices}\
         --batch_size=${BATCH_SIZE} --model=${model}\
         --variable_update=distributed_replicated\
         --num_epochs=${TRAIN_EPOCHS} \
         --job_name=${job_name} --ps_hosts=127.0.0.1:53400 --worker_hosts="${worker_hosts}"\
         --train_dir=${TRAIN_DIR} --task_index=${task_index} --print_training_accuracy=True "${new_args[@]}" &
        echo "Distributed training PID ($!) on device ${device} where job name = ${job_name} and task_index = ${task_index}"
        pid_list+=($!)
    fi
    let i++
done

echo "All background subprocess: ${pid_list[*]}"
ctrl_c
exit ${EXIT_STATUS}
