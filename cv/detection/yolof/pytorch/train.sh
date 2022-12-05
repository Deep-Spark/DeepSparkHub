# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# CONFIG=$1

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:2}

python3 train.py configs/yolof/yolof_r50_c5_8x8_1x_coco.py