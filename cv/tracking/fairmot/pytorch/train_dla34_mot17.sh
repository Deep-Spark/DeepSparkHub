#!/usr/bin/env bash

python3 src/train.py mot --exp_id mot17_dla34 --data_cfg './src/lib/cfg/mot17.json' "$@"
exit $?
