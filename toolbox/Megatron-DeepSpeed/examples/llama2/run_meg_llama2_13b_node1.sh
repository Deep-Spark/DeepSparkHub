#!/bin/bash
# 该脚本可在MRv100 4.0.0的环境中性能达到tokens per second per device: 90.50
# Please change the following envrioment variables
# base on the cluster configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1


PROJ_HOME=$(dirname $(dirname "$PWD"))

DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer.model

CHECKPOINT_PATH=./checkpoints/llama2
mkdir -p $CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH



GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=8080
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --train-iters 250000 \
    --eval-iters 10 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2\
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --disable-bias-linear \
    --use-flash-attn
    --eval-interval 1000 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 20 \
    --make-vocab-size-divisible-by 1

"

MIXED_PRECISION_ARGS="
    --bf16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --attention-softmax-in-fp32 \
    --no-query-key-layer-scaling
"
# --accumulate-allreduce-grads-in-fp32

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-impl mmap \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 98,2,0
"

NETWORK_ARGS="
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --num-key-value-heads 40
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization RMSNorm \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion
"
## group attntion parameters for megatron-lm
## example llama2-70B
# --num-attention-heads 64
# --group-query-attention
# --num-query-groups 8

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

