
source ../_utils/get_num_devices.sh

CURRENT_DIR=$(cd `dirname $0`; pwd)
source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=3}

nonstrict_mode_args=""
if [ "${RUN_MODE}" != "strict" ]; then
    nonstrict_mode_args="--num_epochs 1 --num_iters 300"
fi

ROOT_DIR=${CURRENT_DIR}/../..
DADASAT_PATH=${ROOT_DIR}/data/datasets/MOT17

cd ${ROOT_DIR}/cv/multi_object_tracking/fairmot/pytorch/

bash train_hrnet32_mot17.sh --batch_size $((IX_NUM_CUDA_VISIBLE_DEVICES*BATCH_SIZE)) \
 --lr 0.001 \
 --gpus $(seq -s "," 0 $(($IX_NUM_CUDA_VISIBLE_DEVICES-1))) ${nonstrict_mode_args} --target_loss 20 "$@";check_status

exit ${EXIT_STATUS}
