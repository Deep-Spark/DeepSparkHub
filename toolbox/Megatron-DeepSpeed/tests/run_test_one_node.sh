python3 tests.py \
--timeout_per_case 120 \
--files  'unit_tests/test_utils.py \
unit_tests/test_basic.py \
unit_tests/test_parallel_state.py \
unit_tests/tensor_parallel/test_tensor_parallel_utils.py' \
--master_addr localhost \
--master_port 5673 \
--nnodes 1 \
--node_rank 0
status=$?
if [ $status == 255 ]; then
    exit -1
else
    exit $status
fi
