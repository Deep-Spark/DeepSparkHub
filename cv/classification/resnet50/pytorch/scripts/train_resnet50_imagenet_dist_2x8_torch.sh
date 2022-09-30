# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
SCRIPT_DIR=$(cd `dirname $0`; pwd)
PROJECT_DIR=$SCRIPT_DIR/..
OUTPUT_PATH="$PROJECT_DIR/results"

source $SCRIPT_DIR/get_num_devices.sh

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd $PROJECT_DIR
python3 -m torch.distributed.launch --master_addr ${HOST_MASTER_ADDR} \
	--master_port ${HOST_MASTER_PORT} \
	--nnodes ${HOST_NNODES} \
	--node_rank ${HOST_NODE_RANK} \
	--nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES --use_env \
	train.py \
	--model resnet50 \
	--epochs 90 \
	--acc-thresh 75.9 \
	--output-dir ${OUTPUT_PATH} \
	"$@";check_status

exit ${EXIT_STATUS}
