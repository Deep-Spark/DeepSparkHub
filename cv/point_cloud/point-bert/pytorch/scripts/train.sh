#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

set -x
GPUS=$1

PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=${GPUS} python3 main.py ${PY_ARGS}
