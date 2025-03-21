#!/bin/bash

# Please change the following envrioment variables base on the cluster configuration
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_USE_DIRECT=1

# export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1

PROJ_HOME=$(dirname $(dirname "$PWD"))

DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M_llama3/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer_llama3.model

CHECKPOINT_PATH=./checkpoints/llama2
mkdir -p $CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH

GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=8080
NNODES=1
NODE_RANK=0

TRANSFORMER_IMPL=local

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

## llama3-8B 与 llama2-7B之间有差别的参数
    # --group-query-attention \
    # --num-query-groups 8 \
    # --seq-length 8192 \
    # --max-position-embeddings 8192 \
    # --rotary-position-embeddings-theta 500000 \
    # --ffn-hidden-size 14336 \
    # --tokenizer-type Llama3Tokenizer \
    # vocab_size=128256 不用在脚本里设置

TRAINING_ARGS="
    --train-iters 250000 \
    --eval-iters 10 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --disable-bias-linear \
    --use-flash-attn \
    --eval-interval 1000 \
    --transformer-impl $TRANSFORMER_IMPL \
    --no-fp8-wgrad \
    --use-distributed-optimizer \
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --attention-softmax-in-fp32 \
    --no-query-key-layer-scaling
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 949,50,1
"

NETWORK_ARGS="
    --num-layers 16 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --ffn-hidden-size 14336 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --rotary-position-embeddings-theta 500000 \
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 1234 
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000
"

CHECKPOINTING_ARGS="
    --save-interval 10000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"

LOGGING_ARGS="
    --log-interval 1 \
"
    # --wandb-save-dir $WB_PATH \
    # --tensorboard-dir $TB_PATH \
    # --tensorboard-log-interval 1 

cmd="torchrun $DISTRIBUTED_ARGS $PROJ_HOME/pretrain_gpt_megatron.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS | tee ${LOG_PATH}/output.log 2>&1
    "
echo $cmd
eval $cmd