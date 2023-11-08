# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

CONFIG=$1
GPUS=$2
WORKDIR=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --work-dir $WORKDIR
