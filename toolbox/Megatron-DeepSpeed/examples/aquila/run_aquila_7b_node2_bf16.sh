#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex
export NCCL_SOCKET_IFNAME="ens5f0"

PROJECT_PATH=$(dirname $(dirname "$PWD"))
DS_CONFIG=ds_zero1_config.json
DATA_PATH=${PROJECT_PATH}/dataset/BookCorpusDataset/BookCorpusDataset_text_document 
CHECKPOINT_PATH=./checkpoints/aquila_7b

host_name=$HOST_NAME
addr_array=(${ADDR_ARRAY//,/ }) ## get ip array, split ip str by ','

container_name=$CONTAINER_NAME

HOST_IP=$(hostname -I)
CURRENT_DIR=`pwd`
CUR_SCR=$0
MASTER_PORT=7655

NNODES=2
GPUS_PER_NODE=8
TP=4
PP=2
ZERO_STAGE=1


HIDDEN_SIZE=4096
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096
NUM_KV_HEADS=32

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32 
TRAIN_STEPS=250000 
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

VOCAB_FILE=./tokenizer/vocab.json
MERGE_FILE=./tokenizer/merges.txt
SPECIAL_TOKENS_FILE=./tokenizer/special_tokens.txt

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --tokenizer-type AquilaTokenizer \
    --data-impl mmap \
    --split 1
"

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
  "data_types": {"grad_accum_dtype": "fp32"},
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

OUTPUT_DIR=train_logs/aquila-7b
mkdir -p $OUTPUT_DIR


megatron_args="\
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
      $DATA_ARGS
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
       --eval-interval 2000 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization LayerNorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --no-gradient-accumulation-fusion \
       --use-flash-attn \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1"

function exec_ssh_by_master
{
	# only at master host, start all other non master hosts run
  if [[ "$HOST_IP" =~ "${addr_array[0]}" ]]
	then
		for i in "${!addr_array[@]}"
		do
			if [ "$i" != "0" ]
			then	
				scp ${CUR_SCR} ${host_name}@${addr_array[$i]}:${CURRENT_DIR}
				scp ${CURRENT_DIR}/${DS_CONFIG} ${host_name}@${addr_array[$i]}:${CURRENT_DIR}/${DS_CONFIG}
				ssh ${host_name}@${addr_array[$i]} "docker exec ${container_name} bash -c \"cd ${CURRENT_DIR}; export ADDR_ARRAY=$ADDR_ARRAY; bash ${CUR_SCR} \"" &
			fi
		done
	fi
}

function run_ddp_mm()
{
    for i in "${!addr_array[@]}"
    do
	    if [[ "$HOST_IP" =~ "${addr_array[$i]}" ]]
	    then
		    echo "nodes: ${#addr_array[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${addr_array[0]}"
		    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $i --master_addr ${addr_array[0]} --master_port $MASTER_PORT"
                    torchrun $DISTRIBUTED_ARGS $PROJECT_PATH/pretrain_gpt.py \
			    ${megatron_args} $ds_args | tee ${OUTPUT_DIR}/output.log 2>&1
	    fi
    done
}

exec_ssh_by_master
run_ddp_mm
