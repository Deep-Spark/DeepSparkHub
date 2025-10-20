#!/bin/bash

# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

get_lscpu_value() {
    awk -F: "(\$1 == \"${1}\"){gsub(/ /, \"\", \$2); print \$2; found=1} END{exit found!=1}"
}
lscpu_out=$(lscpu)

n_sockets=$(get_lscpu_value 'Socket(s)' <<< "${lscpu_out}")
n_cores_per_socket=$(get_lscpu_value 'Core(s) per socket' <<< "${lscpu_out}")

echo "Number of CPU sockets on a node: ${n_sockets}"
echo "Number of CPU cores per socket: ${n_cores_per_socket}"

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

export PYTHONPATH=../:$PYTHONPATH

python3 -u -m bind_pyt \
    --nsockets_per_node ${n_sockets} \
    --ncores_per_socket ${n_cores_per_socket} \
    --no_hyperthreads  \
    --no_membind "$@" --training_script ./run_pretraining.py --do_train; check_status

exit ${EXIT_STATUS}