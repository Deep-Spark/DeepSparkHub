# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
#!/bin/bash

GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main.py 2>&1 | tee ./log/cosface_trainlog_`date +%Y%m%d%H%M`.log