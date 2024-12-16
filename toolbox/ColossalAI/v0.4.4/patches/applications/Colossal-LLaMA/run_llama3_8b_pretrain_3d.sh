#!/bin/bash
# set_n_least_used_CUDA_VISIBLE_DEVICES() {
#     local n=${1:-"9999"}
#     echo "GPU Memory Usage:"
#     local FIRST_N_GPU_IDS=$(ixsmi --query-gpu=memory.used --format=csv |
#         tail -n +2 |
#         nl -v 0 |
#         tee /dev/tty |
#         sort -g -k 2 |
#         awk '{print $1}' |
#         head -n $n)
#     export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
#     echo "Now CUDA_VISIBLE_DEVICES is set to:"
#     echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# }

# set_n_least_used_CUDA_VISIBLE_DEVICES 8

PARENT_SAVE_DIR="checkpoint"
PARENT_TENSORBOARD_DIR="tensorboard"
PARENT_CONFIG_FILE="config"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG_DIR="logs/${TIMESTAMP}"
SAVE_DIR="${LOG_DIR}/${PARENT_SAVE_DIR}"
TENSORBOARD_DIR="${LOG_DIR}/${PARENT_TENSORBOARD_DIR}"
CONFIG_FILE="${LOG_DIR}/${PARENT_CONFIG_FILE}.json"

DATASET_PATH=./dataset/llama3_data/arrow/
TOKENIZER_DIR=/home/model_zoos/Meta-Llama-3-8B
GLOBAL_BATCH_SIZE_PER_DP=8
MICRO_BATCH_SIZE=1


mkdir -p $LOG_DIR
colossalai run --nproc_per_node 16 train.py \
    --config "llama3_8b" \
    --dataset  $DATASET_PATH \
    --tokenizer_dir $TOKENIZER_DIR \
    --max_length 8192 \
    --plugin "3d" \
    --zero_stage 1 \
    --pp 4 \
    --custom_ckpt \
    --custom_recompute_layers_per_stage 8 7 6 7 \
    --ignore_steps 2 \
    --use_ixformer_mlp \
    --use_ixformer_fusedrmsnormres \
    --save_interval 0 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 1 \
    --batch_size $GLOBAL_BATCH_SIZE_PER_DP \
    --microbatch_size $MICRO_BATCH_SIZE \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --use_flash_attn \
    --pad_token "eos" |& tee ${LOG_DIR}/output.log

