# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ../_utils/get_num_devices.sh

CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}/../..
DATA_DIR=${ROOT_DIR}/data/datasets/kits19/train
RESUME=${ROOT_DIR}/data/model_zoo/unet3d/model_3620_stage3_start.pth

SEED=1234
MAX_EPOCHS=4200
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=3640
EVALUATE_EVERY=5
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
: ${BATCH_SIZE:=4}
GRADIENT_ACCUMULATION_STEPS=1
SAVE_CKPT="./ckpt_stage3"
LOG_NAME='train_log_stage3.json'

cd ../../../cv/semantic_segmentation/unet3d/pytorch
if [ ! -d ${SAVE_CKPT} ]; then
    mkdir ${SAVE_CKPT};
fi

python3 -u -m torch.distributed.launch --nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES \
main.py --data_dir ${DATA_DIR} \
--epochs ${MAX_EPOCHS} \
--evaluate_every ${EVALUATE_EVERY} \
--start_eval_at ${START_EVAL_AT} \
--quality_threshold ${QUALITY_THRESHOLD} \
--batch_size ${BATCH_SIZE} \
--optimizer sgd \
--ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
--learning_rate ${LEARNING_RATE} \
--seed ${SEED} \
--lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
--output-dir ${SAVE_CKPT} \
--log_name ${LOG_NAME} \
--resume ${RESUME}
"$@"

if [ $? -eq 0 ];then
    echo 'converged to the target value 0.908 of epoch 3820 in full train, stage-wise training succeed'
    exit 0
else
    echo 'not converged to the target value, training fail'
    exit 1
fi

