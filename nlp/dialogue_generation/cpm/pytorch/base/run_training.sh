#!/bin/bash

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
echo $PYTHONPATH

python3 -u -m bind_pyt \
    --nsockets_per_node ${n_sockets} \
    --ncores_per_socket ${n_cores_per_socket} \
    --no_hyperthreads  \
    --no_membind "$@" --training_script ./run_pretraining.py --do_train; check_status

exit ${EXIT_STATUS}