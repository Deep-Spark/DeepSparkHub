source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=16}

CURTIME=`date --utc +%Y%m%d%H%M%S`
CURRENT_DIR=$(cd `dirname $0`; pwd)
MODEL_NAME=`basename ${CURRENT_DIR}`

ROOT_DIR="${CURRENT_DIR}/../.."
DATASET_PATH="${ROOT_DIR}/data/datasets/imagenette"
MODEL_ZOO="${ROOT_DIR}/data/model_zoo"
WORKSPACE="${ROOT_DIR}/output/${MODEL_NAME}/$0/${CURTIME}"
SRC_DIR="${ROOT_DIR}/../models/cv/detection/ssd/tensorflow"

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd ${SRC_DIR}


if [[ -d "./logs" ]]; then
    rm -rf ./logs
fi

: ${CUDA_VISIBLE_DEVICES:="0"}
CUDA_VISIBLE_DEVICES=(${CUDA_VISIBLE_DEVICES//,/ })
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES[0]}

ixdltest-check --nonstrict_mode_args="--train_epochs ${NONSTRICT_EPOCH}" -b 0. --run_script \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONPATH=$PYTHONPATH:${SRC_DIR} python3 train_ssd.py --batch_size ${BATCH_SIZE} --multi_gpu=False \
  --use_amp "$@";  check_status


cd -
exit ${EXIT_STATUS}
