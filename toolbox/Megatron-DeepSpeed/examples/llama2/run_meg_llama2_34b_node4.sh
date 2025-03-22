#!/bin/bash
set -ex
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_NET_SHARED_BUFFERS=0
# export NCCL_DEBUG=INFO
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1


## torch tp overlap
# export ENABLE_TORCH_TP_OVERLAP=1
# export TORCH_TP_OVERLAP_SIZE=4
# export NCCL_USE_HIGHPRIORITYWARP=1
# export NCCL_FORCESYNC_DISABLE=1
# export NCCL_USE_DIRECT=1
# export OMP_NUM_THREADS=4
# export UMD_CCLINLASTCE=1

HOST_NAME="jun.zhao"

ADDR_ARRAY=("10.113.2.10" "10.113.2.9" "10.113.2.11" "10.113.2.12")
CONTAINER_NAME="llama_0323"

HOST_IP=$(echo $(hostname -I) | cut -d " " --output-delimiter="," -f 1)
CURRENT_DIR=`pwd`
CUR_SCR=$0

PROJ_HOME=$(dirname $(dirname "$PWD"))

DATA_PATH=${PROJ_HOME}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer.model

CHECKPOINT_PATH=./checkpoints/llama2
mkdir -p $CHECKPOINT_PATH

DATE=`date +%y%m%d%H%M%S`
LOG_PATH=./logs/$DATE
mkdir -p $LOG_PATH

GPUS_PER_NODE=16
NODES=4

TRANSFORMER_IMPL=local

TRAINING_ARGS="
    --train-iters 250000 \
    --eval-iters 10 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 8\
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --use-flash-attn \
    --sequence-parallel \
    --eval-interval 1000 \
    --transformer-impl $TRANSFORMER_IMPL\
    --use-distributed-optimizer \
    --recompute-granularity full \
    --recompute-method block \
    --make-vocab-size-divisible-by 1 \
    --recompute-num-layers 1 \
    --recompute-method-per-stage 8 1 \
    --recompute-num-layers-per-stage 1 4 1 3 2 2 4 0 \
"
    # --custom-recompute-layers-per-stage 2 2 1 0 0 0 0 0 \
    # --no-gradient-accumulation-fusion \

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
    --num-layers 48 \
    --hidden-size 8192 \
    --ffn-hidden-size 22016 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization RMSNorm \
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

megatron_args="$TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS"

function exec_ssh_by_master
{
	# only at master host, start all other non master hosts run
	if [[ "$HOST_IP" == "${ADDR_ARRAY[0]}" ]]
	then
		for i in "${!ADDR_ARRAY[@]}"
		do
			if [ "$i" != "0" ]
			then
				scp ${CUR_SCR} ${HOST_NAME}@${ADDR_ARRAY[$i]}:${CURRENT_DIR}
                # scp -r ${DATA_PATH} ${HOST_NAME}@${ADDR_ARRAY[$i]}:${DATA_PATH}/../

				ssh ${HOST_NAME}@${ADDR_ARRAY[$i]} "docker exec ${CONTAINER_NAME} bash -c \"cd ${CURRENT_DIR}; bash ${CUR_SCR} \"" &
			fi
		done
	fi
}
function run_ddp_mm()
{
    for i in "${!ADDR_ARRAY[@]}"
    do
	    if [[ "$HOST_IP" == "${ADDR_ARRAY[$i]}" ]]
	    then
		    echo "nodes: ${#ADDR_ARRAY[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${ADDR_ARRAY[0]}"
		    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NODES --node_rank $i --master_addr ${ADDR_ARRAY[0]} --master_port 54321"
                    torchrun $DISTRIBUTED_ARGS $PROJ_HOME/pretrain_gpt_megatron.py \
			    ${megatron_args} | tee ${LOG_PATH}/output.log 2>&1
	    fi
    done
}
exec_ssh_by_master
run_ddp_mm