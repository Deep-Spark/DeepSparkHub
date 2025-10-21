#!/bin/bash

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NUM_GPUS=$1
CONFIG_FILE=$2
PEFT=$3
MODELS_DIR="$PWD/models"
CHECKPOINT_DIR="$PWD/checkpoint"

# 把模型结构拷贝到对应预训练文件路径，方便加载时使用
for source_dir in "$MODELS_DIR"/*; do
    if [ -d "$source_dir" ]; then
        source_dir_name=$(basename "$source_dir")

        for dest_dir in "$CHECKPOINT_DIR"/*; do
            if [ -d "$dest_dir" ]; then
                dest_dir_name=$(basename "$dest_dir")

                if [[ "$dest_dir_name" == *"$source_dir_name"* ]]; then
                    echo "Copying files from $source_dir to $dest_dir"
                    cp -r "$source_dir"/* "$dest_dir"
                fi
            fi
        done
    fi
done


echo "==> Training with $NUM_GPUS gpus | config=$CONFIG_FILE | peft=$PEFT"

torchrun --master_port $MASTER_PORT --nproc_per_node=$NUM_GPUS main.py --train_args_file $CONFIG_FILE --peft_type $PEFT

# for example
# bash run_peft.sh 1 configs/llama2-sft-qlora.json qlora
