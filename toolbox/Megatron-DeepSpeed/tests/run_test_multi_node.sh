# python3 tests.py \
# --timeout_per_case 120 \
# --ignore_timeout \
# --files  'unit_tests/test_utils.py \
# unit_tests/test_basic.py \
# unit_tests/test_parallel_state.py ' \
# --excludes 'unit_tests/tensor_parallel/test_tensor_parallel_utils.py'
# exit $?

## 使用sh脚本将每个ci测试的文件在不同节点上执行
host_name=$HOST_NAME
addr_array=$ADDR_ARRAY
container_name=$CONTAINER_NAME

addr_array=(${ADDR_ARRAY//,/ }) ## get ip array
# addr_array=("10.113.2.1" "10.113.2.2")

HOST_IP=$(hostname -I)
CURRENT_DIR=`pwd`
CUR_SCR=$0
MASTER_PORT=8294
PROJECT_DIR=$(dirname "$PWD")

function exec_ssh_by_master
{
	# only at master host, start all other non master hosts run
	if [[ "$HOST_IP" =~ "${addr_array[0]}" ]]
	then
		for i in "${!addr_array[@]}"
		do
			if [ "$i" != "0" ]
			then	
				
				scp -r ${PROJECT_DIR} ${host_name}@${addr_array[$i]}:$(dirname "$PROJECT_DIR") ## scp whole megatron-deepspeed dir
				ssh ${host_name}@${addr_array[$i]} "docker exec ${container_name} bash -c \"cd ${CURRENT_DIR}; export ADDR_ARRAY=$ADDR_ARRAY; bash ${CUR_SCR} \"" &
			fi
		done
	fi
}

function run_ddp_mm()
{
    for i in "${!addr_array[@]}"
    do
	    if [[ "$HOST_IP" =~ "${addr_array[$i]}" ]]
	    then
		    echo "nodes: ${#addr_array[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${addr_array[0]}"
		    python3 tests.py \
            --master_addr ${addr_array[0]} \
            --master_port $MASTER_PORT \
            --nnodes ${#addr_array[@]} \
            --node_rank $i \
            --timeout_per_case 120 \
            --files  'unit_tests/test_utils.py \
			unit_tests/test_basic.py \
			unit_tests/test_parallel_state.py \
			unit_tests/tensor_parallel/test_tensor_parallel_utils.py'
			status=$?
	    fi
    done
}

exec_ssh_by_master
run_ddp_mm
## 保存退出码，回传给父shell
echo $status > exit_code.txt

exit 0