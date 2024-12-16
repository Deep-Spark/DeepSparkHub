#/bin/bash
set -euox pipefail

CUR_DIR=$(pwd)
if [[ ! -f $CUR_DIR/small-117M.train.jsonl ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/small-117M.train.jsonl
fi

PROJ_HOME=$(dirname "$PWD")
SAVE_PATH=./gpt_small_117M_llama3
mkdir -p $SAVE_PATH

TOKENIZER=Llama3Tokenizer
TOKENIZER_PATH=$PROJ_HOME/examples/llama2/tokenizer/tokenizer_llama3.model

python3 $PROJ_HOME/tools/preprocess_data.py \
            --input ./small-117M.train.jsonl \
            --json-keys text \
            --tokenizer-type $TOKENIZER \
            --tokenizer-model $TOKENIZER_PATH \
            --output-prefix $SAVE_PATH/gpt_small_117M \
            --append-eod \
            --workers 32

rm -f small-117M.train.jsonl