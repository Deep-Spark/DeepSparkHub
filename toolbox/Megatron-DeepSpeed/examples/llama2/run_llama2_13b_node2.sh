#!/bin/bash
set -ex

export NCCL_NET=IB
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_DEBUG=INFO

HOST_NAME="poweruser"

ADDR_ARRAY=("10.113.2.9" "10.113.2.10")
CONTAINER_NAME="llama2"

HOST_IP=$(hostname -I)
CURRENT_DIR=`pwd`
CUR_SCR=$0
LOG_DIR=./train_logs
mkdir -p ${LOG_DIR}

mkdir -p ./config
DS_CONFIG=./config/ds_config.json
PROJECT_PATH=$(dirname $(dirname "$PWD"))
DATA_PATH=${PROJECT_PATH}/dataset/gpt_small_117M/gpt_small_117M_text_document
TOKENIZER_PATH=./tokenizer/tokenizer.model # offical llama tokenizer.model, 默认 tokenizer.vocab_size=32000

# Disabling tensor/pipeline parallelism
TP=4
PP=8

# Model: LLaMA2 - 13B
NLAYERS=40
HIDDEN=5120
FFN_HIDDEN=13824
HEADS=40
SEQ=4096
NUM_KV_HEAD=40

MICRO_BATCH=1
GLOBAL_BATCH_SIZE=32 # e.g. llama: 4M tokens
NODES=2
GPN=16
TRAIN_STEPS=5

ZERO_STAGE=1

# For 1T model, start with microbatch=1, try to get 2 and 4.  If OOM w/ 4, use cpu-offloading
# Set to cpu for offloading to cpu for larger models
# OFFLOAD_DEVICE="cpu"
# CPU_OPTIM=" --cpu-optimizer"

# Set to none and empty string for no cpu offloading
OFFLOAD_DEVICE="none"  
CPU_OPTIM=" "

activation_checkpoint="false"
flash_attention="true"
sequence_parallel="false"


DATE=`date +%m%d%H%M%S`
OUTPUT_DIR=${LOG_DIR}/llama2-70b-nodes${NODES}_mb${MICRO_BATCH}_gbs${GLOBAL_BATCH_SIZE}_TP_${TP}_PP_${PP}_${DATE}
mkdir -p $OUTPUT_DIR


cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "stage3_max_live_parameters": 3e9,
    "stage3_max_reuse_distance": 3e9,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_prefetch_bucket_size": 5e7,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 90000000,
    "sub_group_size": 1e9,
    "offload_optimizer": {
      "device": "$OFFLOAD_DEVICE",
      "buffer_count": 4,
      "pipeline_read": false,
      "pipeline_write": false,
      "pin_memory": true
    }
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
  },
  "wall_clock_breakdown": true,
  "zero_allow_untested_optimizer": false,
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "single_submit": false,
    "overlap_events": true,
    "thread_count": 2
  }
}
EOT


ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ "$PP" == "1" ]
then
    ds_args=" --no-pipeline-parallel ${ds_args}"  # for pipeline parallel
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
fi

megatron_args=" \
           --tensor-model-parallel-size $TP \
           --pipeline-model-parallel-size $PP \
           --num-layers $NLAYERS \
           --hidden-size $HIDDEN \
           --ffn-hidden-size $FFN_HIDDEN \
           --num-attention-heads $HEADS \
           --micro-batch-size $MICRO_BATCH \
           --global-batch-size $GLOBAL_BATCH_SIZE \
           --seq-length $SEQ \
           --max-position-embeddings $SEQ \
           --train-iters ${TRAIN_STEPS} \
           --data-path $DATA_PATH \
           --data-impl mmap \
           --tokenizer-type Llama2Tokenizer \
           --tokenizer-model $TOKENIZER_PATH \
           --split 98,2,0 \
           --lr 3.0e-4 \
           --min-lr 3.0e-5 \
           --lr-decay-style cosine \
           --weight-decay 0.1 \
           --clip-grad 1.0 \
           --adam-beta1 0.9 \
           --adam-beta2 0.95 \
           --log-interval 1 \
           --eval-iters 1 \
           --eval-interval 1000 \
           --save-interval 1000 \
           --bf16 \
           --no-query-key-layer-scaling \
           --attention-dropout 0 \
           --hidden-dropout 0 \
           --use-rotary-position-embeddings \
           --untie-embeddings-and-output-weights \
           --swiglu \
           --normalization RMSNorm \
           --disable-bias-linear \
           --num-key-value-heads $NUM_KV_HEAD \
           --make-vocab-size-divisible-by 1 \
           --exit-interval 5000 \
           --no-gradient-accumulation-fusion \
           --no-masked-softmax-fusion"

if [ "${activation_checkpoint}" = "true" ]; then
    megatron_args="${megatron_args} --checkpoint-activations"
fi

# set flash attention
if [ "${flash_attention}" = "true" ]; then
    megatron_args="${megatron_args} --use-flash-attn"
fi

# set sequence parallel
if [ "$TP" = "1" ]
then
    megatron_args="${megatron_args}"
else
    if [ "${sequence_parallel}" = "true" ];then
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        megatron_args="${megatron_args} --sequence-parallel"
    fi
fi

function exec_ssh_by_master
{
	# only at master host, start all other non master hosts run
	if [[ "$HOST_IP" =~ "${ADDR_ARRAY[0]}" ]]
	then
		for i in "${!ADDR_ARRAY[@]}"
		do
			if [ "$i" != "0" ]
			then
				scp ${CUR_SCR} ${HOST_NAME}@${ADDR_ARRAY[$i]}:${CURRENT_DIR}
				scp ${CURRENT_DIR}/${DS_CONFIG} ${HOST_NAME}@${ADDR_ARRAY[$i]}:${CURRENT_DIR}/${DS_CONFIG}

				ssh ${HOST_NAME}@${ADDR_ARRAY[$i]} "docker exec ${CONTAINER_NAME} bash -c \"cd ${CURRENT_DIR}; bash ${CUR_SCR} \"" &
			fi
		done
	fi
}

function run_ddp_mm()
{
    for i in "${!ADDR_ARRAY[@]}"
    do
	    if [[ "$HOST_IP" =~ "${ADDR_ARRAY[$i]}" ]]
	    then
		    echo "nodes: ${#ADDR_ARRAY[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${ADDR_ARRAY[0]}"
		    DISTRIBUTED_ARGS="--nproc_per_node $GPN --nnodes $NODES --node_rank $i --master_addr ${ADDR_ARRAY[0]} --master_port 54321"
                    torchrun $DISTRIBUTED_ARGS $PROJECT_PATH/pretrain_gpt.py \
			    ${megatron_args} $CPU_OPTIM $ds_args | tee ${OUTPUT_DIR}/output.log 2>&1
	    fi
    done
}

function run_profile()
{
    for i in "${!ADDR_ARRAY[@]}"
    do
            if [[ "$HOST_IP" =~ "${ADDR_ARRAY[$i]}" ]]
            then
                    echo "nodes: ${#ADDR_ARRAY[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${ADDR_ARRAY[0]}"
                    DISTRIBUTED_ARGS="--nproc_per_node $GPN --nnodes $NODES --node_rank $i --master_addr ${ADDR_ARRAY[0]} --master_port 54321"
                    python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $PROJECT_PATH/pretrain_gpt.py \
			    ${megatron_args} $CPU_OPTIM $ds_args --profile | tee ${OUTPUT_DIR}/output.log 2>&1
		    mv profiling_logs ${OUTPUT_DIR}/
            fi
    done
}

exec_ssh_by_master
run_ddp_mm
#run_profile
