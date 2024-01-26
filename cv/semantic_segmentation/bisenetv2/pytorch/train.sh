# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

NUM_GPUS=$1
CONFIG=$2

python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS tools/train_amp.py --config $CONFIG
