#/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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


if [[ ! -f "./small-117M.train.jsonl" ]]; then
     wget https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/small-117M.train.jsonl
     mv small-117M.train.jsonl gpt_small-117M.train.jsonl
fi

pip3 install nltk

CUR_DIR=$(cd "$(dirname "$0")";pwd)


PROJ_HOME=$(dirname "$PWD")
SAVE_PATH=./gpt_small_117M
mkdir -p $SAVE_PATH

MAX_PROMPT_LENGTH=8000
PAD_ID=0

TOKENIZER=Llama2Tokenizer
TOKENIZER_PATH=$PROJ_HOME/examples/llama2/tokenizer/tokenizer.model

python3 $PROJ_HOME/tools/preprocess_data.py \
            --input ./gpt_small-117M.train.jsonl \
            --json-keys text \
            --tokenizer-type $TOKENIZER \
            --tokenizer-model $TOKENIZER_PATH \
            --output-prefix $SAVE_PATH/gpt_small_117M \
            --workers 32 \
            --pad-2-maxlen $MAX_PROMPT_LENGTH \
            --pad-direction left \
            --pad-id $PAD_ID