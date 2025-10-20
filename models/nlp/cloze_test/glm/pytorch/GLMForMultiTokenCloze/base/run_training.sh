#!/bin/bash
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
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
    --no_membind "$@" --training_script ./run_pretraining.py ; check_status

exit ${EXIT_STATUS}