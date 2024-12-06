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

# 本脚本可以带一个参数或者0个参数，指示llama版本，可为 "llama2" 或者 "llama3"，如果无入参，则默认为llama2

set -euox pipefail
CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

DATA_INPUT_DIRS=$CUR_DIR"/dataset/school_math/convert/"
mkdir -p $DATA_INPUT_DIRS

python3 dataset/convert_data.py

LLAMA_VER=${1:-"llama3"}
echo "LLaMA version:" $LLAMA_VER

if [ $LLAMA_VER == "llama2" ]; then
    # 代码中lable与input的错位需要，loss计算length为4096的sequence。
    MAX_LENGTH=4097
    TOKENIZER_DIR=/home/model_zoos/nlp/Llama-2-7b-hf
    DATA_OUTPUT_DIRS=dataset/school_math/convert/llama2_data_sft
    llama_ver=2

elif [ $LLAMA_VER == "llama3" ]; then
    # 代码中lable与input的错位需要，loss计算length为8192的sequence。
    MAX_LENGTH=8193
    TOKENIZER_DIR=/home/model_zoos/nlp/Meta-Llama-3-8B
    DATA_OUTPUT_DIRS=dataset/school_math/convert/llama3_data_sft
    llama_ver=3
else
   echo "Error LLAMA_VER, please input correct LLaMA version" 
   exit 1
fi

python3 dataset/prepare_sft_dataset.py \
    --data_input_dirs $DATA_INPUT_DIRS \
    --data_output_dirs $DATA_OUTPUT_DIRS \
    --tokenizer_dir $TOKENIZER_DIR \
    --max_length $MAX_LENGTH \
    --llama_version  $llama_ver
