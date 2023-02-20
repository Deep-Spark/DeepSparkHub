# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

#!/bin/bash

set -x

: ${HOROVOD_RUN_ARGS:="--gloo"}

# bash ./reset.sh

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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Training phase
horovodrun -np 8 ${HOROVOD_RUN_ARGS}  python3 ./run_pretraining.py \
	--eval_files_dir=./bert_pretrain_tf_records/eval_data \
	--bert_config_file=./bert_pretrain_tf_ckpt/bert_config.json \
	--input_files_dir=./bert_pretrain_tf_records/train_data \
	--train_batch_size=6  \
    --init_checkpoint=./bert_pretrain_tf_ckpt/model.ckpt-28252 \
	--output_dir=${OUTPUT_DIR} \
	--do_train=True \
	--do_eval=True \
	--is_dist_eval_enabled=False \
	--eval_batch_size=24 \
	--max_eval_steps=100 \
	--max_predictions_per_seq=76 \
	--max_seq_length=512 \
	--num_train_steps=2000 \
	--num_accumulation_steps=4 \
	--num_warmup_steps=0 \
	--save_checkpoints_steps=20000 \
	--learning_rate=5e-5 \
	--horovod --amp --nouse_xla \
	--allreduce_post_accumulation=True \
	--enable_device_warmup=True \
	--samples_between_eval=150000 \
	--stop_threshold=0.72 \
	--samples_start_eval=100 \
	--dllog_path=${OUTPUT_DIR}/bert_dllog.json "$@"

exit_code=$?

rm -rf ${OUTPUT_DIR}/*

date +%m%d%H%M%S >> ${LOG_DIR}/time.log
exit ${exit_code}