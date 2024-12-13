#!/bin/bash

set -x

bash ./reset.sh

DATE=`date +%m%d%H%M%S`

OUTPUT_DIR="./results"
if [[ ! -d ${OUTPUT_DIR} ]]; then
    mkdir -p ${OUTPUT_DIR}
fi

LOG_DIR="./logs/train_log_${DATE}"
if [[ ! -d ${LOG_DIR} ]]; then
    mkdir -p ${LOG_DIR}
fi

date +%m%d%H%M%S >> ${LOG_DIR}/time.log

CUDA_VISIBLE_DEVICES=0 python3 ./run_pretraining.py \
	--output_dir=${OUTPUT_DIR} \
	--do_train=True \
	--do_eval=False \
	--is_dist_eval_enabled=False \
	--eval_batch_size=24 \
	--max_eval_steps=100 \
	--max_predictions_per_seq=76 \
	--max_seq_length=512 \
	--num_train_steps=13206 \
	--num_accumulation_steps=1 \
	--num_warmup_steps=0 \
	--save_checkpoints_steps=1000 \
	--learning_rate=5e-5 \
	--amp --nouse_xla \
	--allreduce_post_accumulation=True \
	--enable_device_warmup=False \
	--samples_between_eval=150000 \
	--stop_threshold=0.72 \
	--samples_start_eval=3000000 \
	--dllog_path=${OUTPUT_DIR}/bert_dllog.json "$@"

rm -rf ${OUTPUT_DIR}/*

date +%m%d%H%M%S >> ${LOG_DIR}/time.log