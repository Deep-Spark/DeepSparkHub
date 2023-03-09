#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_squad_gpu.sh rank_size"
echo "for example: bash scripts/run_squad_gpu.sh 8"
echo "assessment_method include: [Accuracy]"
echo "=============================================================================================================="



RANK_SIZE=$1
mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
python3 ${PROJECT_DIR}/../run_squad.py  \
    --config_path="../../task_squad_config.yaml" \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --epoch_num=1 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=12 \
    --eval_batch_size=1 \
    --vocab_file_path="./squad/vocab.txt" \
    --save_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="./squad/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1.ckpt" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="./squad/train.tf_record" \
    --eval_json_path="./squad/dev-v1.1.json" \
    --schema_file_path="" 2>&1 | tee squad.txt
