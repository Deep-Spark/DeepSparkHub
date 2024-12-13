#!/usr/bin/env bash

python3 ./src/train.py mot --exp_id mot17_hrnet18 --arch 'hrnet_18' \
 --data_cfg './src/lib/cfg/mot17.json' \
 --load_model '' "$@"

exit $?
