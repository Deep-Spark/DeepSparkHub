source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=128}

IMAGENET_PATH="`pwd`/../../data/datasets/imagenet"

OUTPUT_PATH="`pwd`/work_dir/efficient_b4"
mkdir -p ${OUTPUT_PATH}

source ../_utils/get_num_devices.sh

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd ../../../cv/classification/efficientnet_b4/pytorch

export PYTHONPATH=./:$PYTHONPATH

: "${HOST_MASTER_ADDR:="127.0.0.1"}"
: "${HOST_MASTER_PORT:=20060}"
: "${HOST_NNODES:=1}"
: "${HOST_NODE_RANK:=0}"

extra_params="--epoch ${NONSTRICT_EPOCH}"
if [ "${RUN_MODE}" == "strict" ]; then
    extra_params="--acc-thresh 75.0"
fi

python3 -m torch.distributed.launch --master_addr ${HOST_MASTER_ADDR} \
--master_port ${HOST_MASTER_PORT} \
--nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES \
--nnodes ${HOST_NNODES} \
--node_rank ${HOST_NODE_RANK} \
--use_env \
train.py \
--model efficientnet_b4 \
--data-path ${IMAGENET_PATH} \
--batch-size ${BATCH_SIZE} \
--acc-thresh 75.0 \
--amp \
--output-dir ${OUTPUT_PATH} ${extra_params} \
"$@";check_status


exit ${EXIT_STATUS}

