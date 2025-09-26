source ../_utils/global_environment_variables.sh
source ../_utils/get_num_devices.sh

: ${BATCH_SIZE:=8}

CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}/../..
DATA_DIR=${ROOT_DIR}/data/datasets/VOC2012_sample

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 0 --run_script \
python3 ../../cv/detection/retinanet/pytorch/train.py \
--model retinanet_resnet50_fpn \
--lr 0.01 \
--data-path ${DATA_DIR} \
--batch-size ${BATCH_SIZE} \
--amp "$@";  check_status

exit ${EXIT_STATUS}
