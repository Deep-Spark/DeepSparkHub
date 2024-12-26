#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -euox pipefail

CUR_DIR=$(pwd)
if [[ ! -f $CUR_DIR/small-117M.train.jsonl ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/data/datasets/small-117M.train.jsonl
fi

if [[ ! -f $CUR_DIR/tokenizer.model ]]; then
    wget -O tokenizer.model http://files.deepspark.org.cn:880/deepspark/data/tokenizer/megatron-lm_tokenizer.model
fi

PROJ_HOME=$(dirname "$PWD")
SAVE_PATH=./gpt_small_117M_Mixtral
mkdir -p $SAVE_PATH

TOKENIZER=Llama2Tokenizer
TOKENIZER_PATH=./tokenizer.model

python3 $PROJ_HOME/tools/preprocess_data.py \
            --input ./small-117M.train.jsonl \
            --json-keys text \
            --tokenizer-type $TOKENIZER \
            --tokenizer-model $TOKENIZER_PATH \
            --output-prefix $SAVE_PATH/gpt_small_117M \
            --append-eod \
            --workers 32

rm -f small-117M.train.jsonl
