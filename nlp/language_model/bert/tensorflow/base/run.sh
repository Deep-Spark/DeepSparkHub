#!/bin/bash

set -x
./init.sh

DATE=`date +%m%d%H%M%S`
DATASET_DIR="/datasets/bert_tfrecord"
BERT_DIR="./pretrain_ckpt"
OUTPUT_DIR="./results"
mkdir -p ${OUTPUT_DIR}
BATCH_SIZE=7
LOG_DIR="./logs/train_log_${DATE}"
mkdir -p ${LOG_DIR}

function run_1card_FPS()
{
    export CUDA_VISIBLE_DEVICES=0
    ../../../../tools/reset.sh
    python3 ./run_pretraining.py \
        --input_files_dir=${DATASET_DIR}/train_data \
        --init_checkpoint=${BERT_DIR}/model.ckpt-28252 \
	--eval_files_dir=${DATASET_DIR}/eval_data \
	--output_dir=${OUTPUT_DIR} \
	--bert_config_file=${BERT_DIR}/bert_config.json \
	--do_train=True \
	--do_eval=False \
	--is_dist_eval_enabled=False \
	--train_batch_size=${BATCH_SIZE} \
	--eval_batch_size=24 \
	--max_eval_steps=100 \
	--max_predictions_per_seq=76 \
	--max_seq_length=512 \
	--num_train_steps=1000 \
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
	--dllog_path=${OUTPUT_DIR}/bert_dllog.json > ${LOG_DIR}/bert_1card_batch_${BATCH_SIZE}_fps.log 2>&1
    rm -rf ${OUTPUT_DIR}/*
}

function run_1card_profile()
{
    export CUDA_VISIBLE_DEVICES=0
    export UMD_ENABLEPROFILE=1
    export TF_ENABLE_CUPTI_PROFILER=1
    export TF_CUPTI_PRINT_KERNEL_INFO=1
    export TF_CUPTI_BUF_SIZE_IN_KB=1024
    export CRT_WRAPPROFLAYER=1
    export DRT_WRAPPROFLAYER=1
    ../../../../tools/reset.sh
    python3 ./run_pretraining.py \
        --input_files_dir=${DATASET_DIR}/train_data \
        --init_checkpoint=${BERT_DIR}/model.ckpt-28252 \
        --eval_files_dir=${DATASET_DIR}/eval_data \
        --output_dir=${OUTPUT_DIR} \
        --bert_config_file=${BERT_DIR}/bert_config.json \
        --do_train=True \
        --do_eval=False \
        --is_dist_eval_enabled=False \
        --train_batch_size=${BATCH_SIZE} \
        --eval_batch_size=24 \
        --max_eval_steps=100 \
        --max_predictions_per_seq=76 \
        --max_seq_length=512 \
        --num_train_steps=15 \
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
        --dllog_path=${OUTPUT_DIR}/bert_dllog.json >${LOG_DIR}/bert_1card_batch_${BATCH_SIZE}_profile.log 2>&1

    rm -rf ${OUTPUT_DIR}/*
    unset UMD_ENABLEPROFILE TF_ENABLE_CUPTI_PROFILER TF_CUPTI_PRINT_KERNEL_INFO TF_CUPTI_BUF_SIZE_IN_KB CRT_WRAPPROFLAYER DRT_WRAPPROFLAYER
}

function run_8card_FPS()
{
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    ../../../../tools/reset.sh
    horovodrun -np 8 python3 ./run_pretraining.py \
        --input_files_dir=${DATASET_DIR}/train_data \
        --init_checkpoint=${BERT_DIR}/model.ckpt-28252 \
        --eval_files_dir=${DATASET_DIR}/eval_data \
        --output_dir=${OUTPUT_DIR} \
        --bert_config_file=${BERT_DIR}/bert_config.json \
        --do_train=True \
        --do_eval=False \
        --is_dist_eval_enabled=False \
        --train_batch_size=${BATCH_SIZE} \
        --eval_batch_size=24 \
        --max_eval_steps=100 \
        --max_predictions_per_seq=76 \
        --max_seq_length=512 \
        --num_train_steps=1000 \
        --num_accumulation_steps=1 \
        --num_warmup_steps=0 \
        --save_checkpoints_steps=1000 \
        --learning_rate=5e-5 \
        --horovod --amp --nouse_xla \
        --allreduce_post_accumulation=True \
        --enable_device_warmup=False \
        --samples_between_eval=150000 \
        --stop_threshold=0.72 \
        --samples_start_eval=3000000 \
        --dllog_path=${OUTPUT_DIR}/bert_dllog.json > ${LOG_DIR}/bert_8card_batch_${BATCH_SIZE}_fps.log 2>&1

    rm -rf ${OUTPUT_DIR}/*
    
}

function run_8card_end2end()
{
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    ../../../../tools/reset.sh
    horovodrun -np 8 python3 ./run_pretraining.py \
        --input_files_dir=${DATASET_DIR}/train_data \
        --init_checkpoint=${BERT_DIR}/model.ckpt-28252 \
        --eval_files_dir=${DATASET_DIR}/eval_data \
        --output_dir=${OUTPUT_DIR} \
        --bert_config_file=${BERT_DIR}/bert_config.json \
        --do_train=True \
        --do_eval=False \
        --is_dist_eval_enabled=False \
        --train_batch_size=${BATCH_SIZE} \
        --eval_batch_size=24 \
        --max_eval_steps=100 \
        --max_predictions_per_seq=76 \
        --max_seq_length=512 \
        --num_train_steps=13206 \
        --num_accumulation_steps=4 \
        --num_warmup_steps=0 \
        --save_checkpoints_steps=1000 \
        --learning_rate=5e-5 \
        --horovod --amp --nouse_xla \
        --allreduce_post_accumulation=True \
        --enable_device_warmup=True \
        --samples_between_eval=150000 \
        --stop_threshold=0.72 \
        --samples_start_eval=3000000 \
        --dllog_path=${OUTPUT_DIR}/bert_dllog.json > ${LOG_DIR}/bert_8card_batch_${BATCH_SIZE}_end2end.log 2>&1
    rm -rf ${OUTPUT_DIR}/*
}
date +%m%d%H%M%S >> ${LOG_DIR}/time.log
run_1card_FPS
run_1card_profile
run_8card_FPS
run_8card_end2end
date +%m%d%H%M%S >> ${LOG_DIR}/time.log
