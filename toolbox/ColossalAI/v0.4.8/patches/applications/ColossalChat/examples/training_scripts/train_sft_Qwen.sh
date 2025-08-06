nproc_per_node=4
tp=1
pp=4
batch_size=32

PROJECT_NAME="SFT"
PARENT_SAVE_DIR="train_results/qwen2.5_3B/" # Path to a folder to save checkpoints
PARENT_CONFIG_FILE=$PARENT_SAVE_DIR # Path to a folder to save training config logs
PARENT_LOG_DIR=$PARENT_SAVE_DIR # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/lin.wu/model/Qwen2.5-3B" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH=$PRETRAINED_MODEL_PATH # huggingface or local tokenizer path
declare -a dataset=(
    ../data_preparation_scripts/processed_data/sft/arrow/part-00000
    ../data_preparation_scripts/processed_data/sft/arrow/part-00001
    ../data_preparation_scripts/processed_data/sft/arrow/part-00002
    ../data_preparation_scripts/processed_data/sft/arrow/part-00003
    ../data_preparation_scripts/processed_data/sft/arrow/part-00004
    ../data_preparation_scripts/processed_data/sft/arrow/part-00005
    ../data_preparation_scripts/processed_data/sft/arrow/part-00006
    ../data_preparation_scripts/processed_data/sft/arrow/part-00007
    ../data_preparation_scripts/processed_data/sft/arrow/part-00008
    ../data_preparation_scripts/processed_data/sft/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"
LOG_DIR="${PARENT_LOG_DIR}${FULL_PROJECT_NAME}"

echo $(which colossalai)
echo $(which python)

# ixprof --profile-child-processes \
colossalai run --nproc_per_node $nproc_per_node --master_port 31312 --hostfile ./hostfile train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --save_interval 2000 \
    --dataset ${dataset[@]} \
    --plugin 3d \
    --tp $tp \
    --pp $pp \
    --zero_stage 1 \
    --batch_size $batch_size \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --lr 5e-5 \
    --max_length 4096 \
    --use_flash_attn \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --log_dir $LOG_DIR \
    --custom_ckpt
    # > qwen2_3B_tp${tp}pp${pp}bsz${batch_size}.profile 2>&1