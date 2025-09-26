#!/bin/bash -x

#bash init.sh

DATASET_DIR="/datasets/coco_2017"
BATCH_SIZE_LIST="128"
NHWC_PARAMS="--nhwc --pad-input --jit"
DATE=`date +%m%d%H%M%S`
LOG_DIR="./train_log_${DATE}"
CORES_PER_SOCKET=`lscpu|awk '/Core\(s\) per socket/ {print $4}'`
HOST_IP=$(hostname -I | awk '{print $1}')
CUR_DIR=`pwd`
CUR_SCR=$0

ADDR_ARRAY=("192.168.10.19" "192.168.10.20" "192.168.10.21" "192.168.10.22")
IMAGE="<url>:v2.1.0_0304-143_5.4.0-42"
CONTAINER_NAME="slw_release_2.1_0304"
CONTAINER_INIT_OPT="--privileged --pid=host --cap-add=ALL -v /dev:/dev -v /lib/modules:/lib/modules -v /home/shunlai.wang:/home/shunlai.wang  -v /mnt:/mnt -v /data/datasets:/datasets --network=host -e BASH_ENV=/etc/bash.bashrc"
mkdir -p ${LOG_DIR}

function run_multi_machine_8card_FPS()
{
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	export GLOO_SOCKET_IFNAME="ib0"
	FORMAT=$1
	BACKEND=$2
	num_epochs=10
        if [ "${FORMAT}" = "nchw" ]; then
            args=""
        else
            args=${NHWC_PARAMS}
        fi

	# do actual run when IP matched
	for i in "${!ADDR_ARRAY[@]}"
	do
		if [ "$HOST_IP" == "${ADDR_ARRAY[$i]}" ]
		then
			for BATCH_SIZE in ${BATCH_SIZE_LIST}
			do	
				# ../../../../tools/reset.sh
				echo "nodes: ${#ADDR_ARRAY[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${ADDR_ARRAY[0]}"
				python -u -m bind_launch --nnodes ${#ADDR_ARRAY[@]} --node_rank $i --nproc_per_node 8 --master_addr ${ADDR_ARRAY[0]} --nsockets_per_node 2 --ncores_per_socket ${CORES_PER_SOCKET} --no_membind \
					./train.py --dali --data=${DATASET_DIR} --batch-size=${BATCH_SIZE} --warmup-factor=0 --warmup=650 --lr=5.2e-3 --threshold=0.23 --no-save --epochs ${num_epochs} --eval-batch-size=160 --wd=1.6e-4 --use-fp16 --delay-allreduce --lr-decay-factor=0.2 --lr-decay-epochs 34 45 --opt-level "O2" ${args} --backend ${BACKEND,,} > ${LOG_DIR}/ssd_${#ADDR_ARRAY[@]}_machine_8card_${FORMAT,,}_batch_${BATCH_SIZE}_${BACKEND,,}_fps.log 2>&1
			done
		fi
	done

}


function exec_ssh_by_master() {
# only at master host, start all other non master hosts run
if [ "$HOST_IP" == "${ADDR_ARRAY[0]}" ]
then
        for i in "${!ADDR_ARRAY[@]}"
        do
                if [ "$i" != "0" ]
                then
                        scp ${CUR_SCR} ${ADDR_ARRAY[$i]}:${CUR_DIR}
			init_docker_container_by_non_master ${ADDR_ARRAY[$i]}
                        ssh ${ADDR_ARRAY[$i]} "docker exec -i ${CONTAINER_NAME} bash -c \"cd ${CUR_DIR}; bash ${CUR_SCR} \"" &
                fi
        done
fi

}


function init_docker_container_by_non_master
{
	ADDR=$1
	matched_containers=`ssh ${ADDR} "docker ps -a"|grep "${CONTAINER_NAME}$"|wc -l`
	if [ ${matched_containers} -gt "0" ]
	then
		echo "Warning: Found container ${CONTAINER_NAME} exists! Will delete it."
		ssh ${ADDR} "docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}"
	fi
	ssh ${ADDR} "docker run -itd --name ${CONTAINER_NAME} ${CONTAINER_INIT_OPT} ${IMAGE} /bin/bash"
	if [ "$?" != "0" ]
	then
		echo "Error: Init container ${CONTAINER_NAME} at ${ADDR} failed!"
		exit -1
	fi
}



function run_multi_machine_8card_end2end()
{
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	export GLOO_SOCKET_IFNAME="ib0"
	FORMAT=$1
	BACKEND=$2
	BATCH_SIZE=$3
	num_epochs=90
        if [ "${FORMAT}" = "nchw" ]; then
            args=""
        else
            args=${NHWC_PARAMS}
        fi

	# do actual run when IP matched
	for i in "${!ADDR_ARRAY[@]}"
	do
		if [ "$HOST_IP" == "${ADDR_ARRAY[$i]}" ]
		then
			../../../../tools/reset.sh
			echo "nodes: ${#ADDR_ARRAY[@]}, rank: $i, IP: $HOST_IP, MASTER_IP: ${ADDR_ARRAY[0]}"
			python3 -u -m bind_launch --nnodes ${#ADDR_ARRAY[@]} --node_rank $i --nproc_per_node 8 --master_addr ${ADDR_ARRAY[0]} --nsockets_per_node 2 --ncores_per_socket ${CORES_PER_SOCKET} --no_membind \
                                        ./train.py --dali --data=${DATASET_DIR} --batch-size=${BATCH_SIZE} --warmup-factor=0 --warmup=650 --lr=2.68e-3 --threshold=0.23 --no-save --epochs ${num_epochs} --eval-batch-size=160 --wd=1.6e-4 --use-fp16 --delay-allreduce --lr-decay-factor=0.2 --lr-decay-epochs 34 45 --opt-level "O2" ${args} --backend ${BACKEND,,} > ${LOG_DIR}/ssd_${#ADDR_ARRAY[@]}_machine_8card_${FORMAT,,}_batch_${BATCH_SIZE}_${BACKEND,,}_fps.log 2>&1

		fi

	done
}

date +%m%d%H%M%S >> ${LOG_DIR}/time.log
exec_ssh_by_master
#run_multi_machine_8card_end2end nhwc gloo 128
run_multi_machine_8card_end2end nhwc nccl 112
#run_multi_machine_8card_end2end nhwc gloo 96
#run_multi_machine_8card_end2end nhwc gloo 80
#run_multi_machine_8card_end2end nhwc gloo 64
#run_multi_machine_8card_end2end nhwc gloo 56
date +%m%d%H%M%S >> ${LOG_DIR}/time.log
