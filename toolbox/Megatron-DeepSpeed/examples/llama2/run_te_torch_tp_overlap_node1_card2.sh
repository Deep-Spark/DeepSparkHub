#!/bin/bash

# Please change the following environment variables
# base on the cluster configuration
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=ens5f0

export ENABLE_TORCH_TP_OVERLAP=1
export TORCH_TP_OVERLAP_SIZE=2

# following environment variables must be set when ENABLE_TORCH_TP_OVERLAP=1
export NCCL_FORCESYNC_DISABLE=1
export NCCL_USE_DIRECT=1
export OMP_NUM_THREADS=4
export UMD_CCLINLASTCE=1


PROJ_HOME=$(dirname $(dirname "$PWD"))
DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer.model

CHECKPOINT_PATH=./checkpoints/llama2
mkdir -p $CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH
TRANSFORMER_IMPL=transformer_engine
# TB_PATH=./tboard/$DATE
# mkdir -p $TB_PATH
# WB_PATH=./wandb/$DATE
# mkdir -p $WB_PATH

# Change for multinode config
# export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
# export GPUS_PER_NODE=$(awk '{$1=$1;print}' $HOSTFILE|awk -F" |=" '{ranks[$1]=$NF;}END{print ranks["'$NODE_ADDR'"];}')
# export NNODES=$(awk '{$1=$1;print}' $HOSTFILE | wc -l)
# export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
# export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
# export MASTER_PORT=12346
# WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP=2
PP=1
GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=8081
NNODES=1
NODE_RANK=0


# llama2-7b
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=4
NUM_HEADS=32
SEQ_LENGTH=4096
NUM_KV_HEADS=32

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2

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
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --disable-bias-linear \
    --eval-interval 1000 \
    --use-flash-attn
    --bf16
    --transformer-impl $TRANSFORMER_IMPL\
    --no-gradient-accumulation-fusion \
"
    # --sequence-parallel \
    # --use-distributed-optimizer \

# MIXED_PRECISION_ARGS="
#     --bf16 \
#     --initial-loss-scale 522893 \
#     --min-loss-scale 1.0 \
#     --attention-softmax-in-fp32 
# "
# --accumulate-allreduce-grads-in-fp32


DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 949,50,1
"

NETWORK_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads $NUM_HEADS \
    --num-key-value-heads $NUM_KV_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --norm-epsilon 1e-5 \
    --swiglu \
    --normalization RMSNorm \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights
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
    --attention-dropout 0.3 \
    --hidden-dropout 0.3 \
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
