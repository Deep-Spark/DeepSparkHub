
CURRENT_DIR=$(cd `dirname $0`; pwd)
source ../_utils/get_num_devices.sh
source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=32}

ROOT_DIR=${CURRENT_DIR}/../..
DATA_DIR=${ROOT_DIR}/data/datasets/imagenette

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd $CURRENT_DIR/../../../models/cv/classification/mobilenetv3/pytorch/
ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 10 --run_script \
python3 train.py --model  mobilenet_v3_large --data-path "${DATA_DIR}" \
   --epochs 600 --batch-size ${BATCH_SIZE} --opt sgd --lr 0.1 \
    --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2 \
	--output-dir . --amp "$@"; check_status
exit ${EXIT_STATUS}
