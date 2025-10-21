#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

# SEED=${1:--1}
SEED=1234
MAX_EPOCHS=5000
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=400
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
DATASET_DIR="/home/datasets/cv/kits19/train"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
SAVE_CKPT="./ckpt_full"
LOG_NAME='full_train_log.json'

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
#   python -c "
# from mlperf_logging.mllog import constants
# from runtime.logging import mllog_event
# mllog_event(key=constants.CACHE_CLEAR, value=True)"
#PYTHONPATH=../../../ 
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 -u -m torch.distributed.launch --nproc_per_node=8 main.py --data_dir ${DATASET_DIR} \
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

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi