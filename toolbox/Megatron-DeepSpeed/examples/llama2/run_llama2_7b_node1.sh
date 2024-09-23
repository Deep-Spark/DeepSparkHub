#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex
export NCCL_SOCKET_IFNAME="ens5f0"

PROJECT_PATH=$(dirname $(dirname "$PWD"))
mkdir -p ./config
DS_CONFIG=./config/ds_config.json
DATA_PATH=${PROJECT_PATH}/dataset/gpt_small_117M/gpt_small_117M_text_document
CHECKPOINT_PATH=./checkpoints/llama2
TOKENIZER_PATH=./tokenizer/tokenizer.model # offical llama tokenizer.model, 默认 tokenizer.vocab_size=32000

TP=4
PP=4
ZERO_STAGE=1

GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=8080
NNODES=1
NODE_RANK=0

# llama2-7b
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096
NUM_KV_HEADS=32

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32 # e.g. llama: 4M tokens
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1


cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "data_types": {
    "grad_accum_dtype": "fp32"
  },
  "fp16": {
      "enabled": false,
      "auto_cast": false,
      "loss_scale": 0,
      "initial_scale_power": 16,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

OUTPUT_DIR=train_logs/llama2-7b
mkdir -p $OUTPUT_DIR

torchrun $DISTRIBUTED_ARGS \
       $PROJECT_PATH/pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --no-gradient-accumulation-fusion \
       --use-flash-attn \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       $ds_args | tee ${OUTPUT_DIR}/output.log 2>&1
