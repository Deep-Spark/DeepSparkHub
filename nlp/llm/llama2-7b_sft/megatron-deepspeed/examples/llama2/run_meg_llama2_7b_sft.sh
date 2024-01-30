#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

# Please change the following envrioment variables
# base on the cluster configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_HCA=mlx5_0,mlx5_3
# export NCCL_DEBUG=debug
# export OMP_NUM_THREADS=4

PROJ_HOME=$(dirname $(dirname "$PWD"))

DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer.model

CHECKPOINT_PATH=./checkpoints/llama2
LOAD_CHECKPOINT_PATH=${PROJ_HOME}/checkpoints/llama2_7b_megatron
mkdir -p $CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH

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

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=8084
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
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --disable-bias-linear \
    --eval-interval 1000 \
    --no-gradient-accumulation-fusion \
"

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
    --attention-softmax-in-fp32 \
    --no-query-key-layer-scaling
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 949,50,1
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights
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
    --load $LOAD_CHECKPOINT_PATH
        --use-checkpoint-args \
        --no-load-optim \
        --no-load-rng \
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