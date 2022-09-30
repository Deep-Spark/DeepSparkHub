# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
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
