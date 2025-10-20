#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_DEBUG=TRACE
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=4
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
export NCCL_USE_DIRECT=1

DATA_PATH=datasets/gpt_small_117M_llama3/gpt_small_117M_text_document
TOKENIZER_MODEL=llama3-8b

TP=1
PP=8
NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8
SEQ_LEN=8192
MAX_POSITION_EMBEDDINGS=8192

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --tokenizer-type HuggingFaceTokenizer 
    --tokenizer-model ${TOKENIZER_MODEL}
    --group-query-attention 
    --num-query-groups ${NUM_KEY_VALUE_HEADS}
    --attention-dropout 0.0
    --hidden-dropout 0.0 
    --attention-softmax-in-fp32 
    --normalization RMSNorm 
    --position-embedding-type rope 
    --rotary-base 500000 
    --rotary-percent 1.0
    --untie-embeddings-and-output-weights 
    --disable-bias-linear 
    --transformer-impl transformer_engine 
    --swiglu 
    --bf16 
    --use-legacy-models 
    --ckpt-format torch
)

TRAINING_ARGS=(
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --weight-decay 0.1 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --min-lr 6.0e-6
    --lr-decay-style cosine 
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --micro-batch-size 1 
    --global-batch-size 64 
    --train-iters 5 
    --init-method-std 0.006 
    --no-load-optim 
    --no-load-rng 
    --no-create-attention-mask-in-dataloader
    --initial-loss-scale 65536 
    --use-flash-attn 
    --num-layers-per-stage 1 3 4 4 2 5 1 3 
    --recompute-granularity=full
    --recompute-method=uniform 
    --recompute-num-layers 1 
    --recompute-method-per-stage 8 1 
    --recompute-num-layers-per-stage 5 2 3 0 
    --use-distributed-optimizer 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP} 
    --pipeline-model-parallel-size ${PP}
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --split 99,1,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 0
)

torchrun ${DISTRIBUTED_ARGS[@]} ./pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}