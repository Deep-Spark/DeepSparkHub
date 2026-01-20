CURRENT_DIR=$(cd `dirname $0`; pwd)

source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=8}

ROOT_DIR=${CURRENT_DIR}/../..

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd "${ROOT_DIR}/deepsparkhub-gpl/cv/detection/yolov5/pytorch"
ixdltest-check --nonstrict_mode_args="--epoch 1" -b 0.2 --run_script \
python3 train.py --img-size 640 --batch-size ${BATCH_SIZE} \
 --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --data ./data/coco.yaml  --amp ${nonstrict_mode_args} "$@";  check_status

exit ${EXIT_STATUS}
