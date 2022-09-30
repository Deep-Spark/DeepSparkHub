#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node ${GPU_NUMS}\
 src/train.py mot --exp_id mot17_dla34\
 --data_cfg './src/lib/cfg/mot17.json' "$@"
