#!/bin/bash
# 本脚本可以带一个参数或者0个参数，指示llama版本，可为 "llama2" 或者 "llama3"，如果无入参，则默认为llama2

set -euox pipefail
CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

if [[ ! -f $CUR_DIR/small-117M.train.jsonl ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/data/datasets/small-117M.train.jsonl
fi

DATA_INPUT_DIRS=$CUR_DIR

LLAMA_VER=${1:-"llama3"}
echo "LLaMA version:" $LLAMA_VER

if [ $LLAMA_VER == "llama2" ]; then
    # 代码中lable与input的错位需要，loss计算length为4096的sequence。
    MAX_LENGTH=4097
    TOKENIZER_DIR=/home/model_zoos/Llama-2-7b-hf
    DATA_OUTPUT_DIRS=dataset/llama2_data

elif [ $LLAMA_VER == "llama3" ]; then
    # 代码中lable与input的错位需要，loss计算length为8192的sequence。
    MAX_LENGTH=8193
    TOKENIZER_DIR=/home/model_zoos/Meta-Llama-3-8B
    DATA_OUTPUT_DIRS=dataset/llama3_data

else
   echo "Error LLAMA_VER, please input correct LLaMA version" 
   exit 1
fi

python3 dataset/prepare_pretrain_dataset.py \
    --data_input_dirs $DATA_INPUT_DIRS \
    --data_output_dirs $DATA_OUTPUT_DIRS \
    --dataset_type webtext \
    --tokenizer_dir $TOKENIZER_DIR \
    --max_length $MAX_LENGTH \
