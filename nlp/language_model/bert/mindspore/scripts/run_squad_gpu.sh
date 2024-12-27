#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
echo "bash scripts/run_squad_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_squad_gpu.sh 1"
echo "assessment_method include: [Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="$1"
fi

mkdir -p squad
# 1. Download training dataset(.tf_record), eval dataset(.json), vocab.txt and checkpoint
training_tf_record_file=squad/train.tf_record
if [[ ! -f "${training_tf_record_file}" ]]; then
  cd squad
  wget http://files.deepspark.org.cn:880/deepspark/data/datasets/squad_data_with_tf_record.tar
  tar xvf squad_data_with_tf_record.tar
  rm -rf squad_data_with_tf_record.tar
  cd -
fi

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python3 ${PROJECT_DIR}/../run_squad.py  \
    --config_path="../../task_squad_config.yaml" \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --device_id=0 \
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
