#/bin/bash

PROJ_HOME=$(dirname "$PWD")
SAVE_PATH=./gpt_small_117M
mkdir -p $SAVE_PATH

TOKENIZER=Llama2Tokenizer
TOKENIZER_PATH=$PROJ_HOME/examples/llama2/tokenizer/tokenizer.model

python3 $PROJ_HOME/tools/preprocess_data.py \
            --input ./gpt_small-117M.train.jsonl \
            --json-keys text \
            --tokenizer-type $TOKENIZER \
            --tokenizer-model $TOKENIZER_PATH \
            --output-prefix $SAVE_PATH/gpt_small_117M \
            --append-eod \
            --workers 32




