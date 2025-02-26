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
# export NCCL_SOCKET_IFNAME=ens5f0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_HCA=mlx5_0,mlx5_3
# export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4


TP=4
PP=4
GLOBAL_BATCH_SIZE=1
INFERENCE_MICRO_BATCH_SIZE=1
TRAIN_MICRO_BATCH_SIZE=1

GPUS_PER_NODE=$(($TP*$PP))
MASTER_ADDR=localhost
MASTER_PORT=8880
NNODES=1
NODE_RANK=0

PROJ_HOME=$(dirname $(dirname "$PWD"))

# DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=${PROJ_HOME}/checkpoints/llama2-7b/tokenizer.model

ACTOR_MODEL_PATH=${PROJ_HOME}/checkpoints/rlhf_llama2_7b_tp${TP}_pp${PP}
CRITIC_MODEL_PATH=${PROJ_HOME}/checkpoints/rlhf_tinyllama_1.1b_tp${TP}_pp${PP}

ACTOR_LR=1e-7
CRITIC_LR=2e-6

ACTOR_WEIGHT_DECAY=0.1
CRITIC_WEIGHT_DECAY=0.1

MAX_PROMPT_SEQ_LEN=8000
MAX_ANSWER_SEQ_LEN=1024

SAVE_CHECKPOINT_PATH=./checkpoints
mkdir -p $SAVE_CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --RLHF \
    --train-iters 250000 \
    --eval-iters 10 \
    --ppo-epoches 1 \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --custom-partition 5 5 6 6 \
    --micro-batch-size ${INFERENCE_MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --rlhf-train-mbs ${TRAIN_MICRO_BATCH_SIZE} \
    --disable-bias-linear \
    --use-flash-attn \
    --eval-interval 1000 \
    --empty-unused-memory-level 0
"
    # --use-distributed-optimizer \

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
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 949,50,1 \
    --max-prompt-seq-len $MAX_PROMPT_SEQ_LEN \
    --decoder-seq-length $MAX_ANSWER_SEQ_LEN
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 1234 
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --actor-weight-decay ${ACTOR_WEIGHT_DECAY} \
    --critic-weight-decay ${CRITIC_WEIGHT_DECAY} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --actor-learning-rate ${ACTOR_LR} \
    --critic-learning-rate ${CRITIC_LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters 10
"

CHECKPOINTING_ARGS="
    --actor_model_name_or_path $ACTOR_MODEL_PATH \
    --critic_model_name_or_path $CRITIC_MODEL_PATH \
    --save-interval 10000 \
    --save $SAVE_CHECKPOINT_PATH \
"

LOGGING_ARGS="
    --log-interval 1 \
"

cmd="torchrun $DISTRIBUTED_ARGS $PROJ_HOME/train_rlhf_llama.py \
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

status=$?
exit $status