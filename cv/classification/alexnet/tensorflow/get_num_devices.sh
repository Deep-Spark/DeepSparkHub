#!/bin/bash

devices=$CUDA_VISIBLE_DEVICES
if [ -n "$devices"  ]; then
    _devices=(${devices//,/ })
    num_devices=${#_devices[@]}
else
    num_devices=2
    export CUDA_VISIBLE_DEVICES=0,1
    echo "Not found CUDA_VISIBLE_DEVICES, set nproc_per_node = ${num_devices}"
fi
export IX_NUM_CUDA_VISIBLE_DEVICES=${num_devices}