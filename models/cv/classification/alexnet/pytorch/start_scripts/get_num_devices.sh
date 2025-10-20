#!/bin/bash

devices=$CUDA_VISIBLE_DEVICES
if [ -n "$devices"  ]; then
    _devices=(${devices//,/ })
    num_devices=${#_devices[@]}
else
    num_devices=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    echo "Not found CUDA_VISIBLE_DEVICES, set nproc_per_node = ${num_devices}"
fi
export IX_NUM_CUDA_VISIBLE_DEVICES=${num_devices}
