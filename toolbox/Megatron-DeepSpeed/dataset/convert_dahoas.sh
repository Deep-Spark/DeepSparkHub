#/bin/bash
CUR_DIR=$(cd "$(dirname "$0")";pwd)

if [[ ! -e ${CUR_DIR}/dahoas_train.jsonl ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/data/datasets/dahoas_train.jsonl
fi

PROJ_HOME=$(dirname "$PWD")
SAVE_PATH=./dahoas
mkdir -p $SAVE_PATH

MAX_PROMPT_LENGTH=16000
PAD_ID=0

TOKENIZER=Llama2Tokenizer
TOKENIZER_PATH=$PROJ_HOME/examples/llama2/tokenizer/tokenizer.model

python3 $PROJ_HOME/tools/preprocess_data.py \
            --input ./dahoas_train.jsonl \
            --json-keys prompt \
            --tokenizer-type $TOKENIZER \
            --tokenizer-model $TOKENIZER_PATH \
            --output-prefix $SAVE_PATH/dahoas_train \
            --workers 32 \
            --pad-2-maxlen $MAX_PROMPT_LENGTH \
            --pad-direction left \
            --pad-id $PAD_ID