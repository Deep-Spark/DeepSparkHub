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
# =================================================
# Constants
# =================================================

export MODEL="cpm"
NEXP=1

WORK_DIR="/workspace/modelzoo/cpm/pytorch"
MODEL_DIR="${WORK_DIR}/base"

CURRENT_DIR=$(cd `dirname $0`; pwd)
PROJ_DIR="${CURRENT_DIR}/../"
BUILD_EXTENSION_DIR="${CURRENT_DIR}/build"
BUILD_EXTENSION_PACKAGE_NAME="ext_ops"

BASE_DOCKERFILE_PATH="${CURRENT_DIR}/BaseDockerfile"
HOST_DOCKERFILE_PATH="${CURRENT_DIR}/Dockerfile"

SOURCE_DATA_DIR=""
MAP_DATA_DIR="/mnt/dataset/modelzoo/${MODEL}"
SUBMITTER="iluvatar"
CONFIG=""

: "${CLEAR_CACHES:=1}"
SHM_SIZE="32g"


# =================================================
# Parse arguments
# =================================================

i=2
TRAINING_SCRIPT_ARGS="$@"
for arg in "$@"
do
    if [[ $arg =~ "--data_dir" ]]; then
        if [[ $arg =~ "=" ]]; then
            kv=(${arg//=/ })
            SOURCE_DATA_DIR=${kv[1]}
            TRAINING_SCRIPT_ARGS=${TRAINING_SCRIPT_ARGS/$arg/"--data_dir ${MAP_DATA_DIR}"}
        else
            SOURCE_DATA_DIR=${!i}
            TRAINING_SCRIPT_ARGS=${TRAINING_SCRIPT_ARGS/"--data_dir ${!i}"/"--data_dir ${MAP_DATA_DIR}"}
        fi

    elif [[ $arg =~ "--name" ]]; then
        if [[ $arg =~ "=" ]]; then
            kv=(${arg//=/ })
            SUBMITTER=${kv[1]}
        else
            SUBMITTER=${!i}
        fi

    elif [[ $arg =~ "--config" ]]; then
        if [[ $arg =~ "=" ]]; then
            kv=(${arg//=/ })
            CONFIG=${kv[1]}
        else
            CONFIG=${!i}
        fi
    fi

    let i++
done


# =================================================
# Check arguments
# =================================================

if [[ "${SOURCE_DATA_DIR}" == "" ]]; then
    echo "ERROR: data_dir is not given, please set --data_dir <DATA_DIR>"
    exit 1
fi

if [[ "${CONFIG}" == "" ]]; then
    echo "ERROR: config is not given, please set --config <CONFIG>"
    exit 1
fi

CONTAINER_SUBMITTER_DIR="${WORK_DIR}/${SUBMITTER}"
HOST_SUBMITTER_DIR="${PROJ_DIR}/${SUBMITTER}"

CONTAINER_ENVIRONMENT_VARIABLES_PATH=${CONTAINER_SUBMITTER_DIR}/${MODEL}/config/environment_variables.sh
HOST_ENVIRONMENT_VARIABLES_PATH="${HOST_SUBMITTER_DIR}/${MODEL}/config/environment_variables.sh"

HOST_SUBMITTER_DOCKERFILE="${PROJ_DIR}/${SUBMITTER}/${MODEL}/config/Dockerfile"

DOCKER_IMAGE="modelzoo-${SUBMITTER}:${MODEL}"
CONTAINER_NAME="modelzoo-${SUBMITTER}_${MODEL}-container"

if [ ! -f "${HOST_ENVIRONMENT_VARIABLES_PATH}" ]; then
    touch "${HOST_ENVIRONMENT_VARIABLES_PATH}"
fi

source ${HOST_ENVIRONMENT_VARIABLES_PATH}

RESULTS_DIR="${PROJ_DIR}/${SUBMITTER}/${MODEL}/results"
LOG_FILE_BASE="${RESULTS_DIR}/config_${CONFIG}_experiment_`date +%s`"

echo "======================================"
echo "Arguments"
echo "---------"

echo "MODEL = ${MODEL}"
echo "CONTAINER_NAME = ${CONTAINER_NAME}"
echo "DOCKER_IMAGE = ${DOCKER_IMAGE}"
echo "MODEL_DIR = ${MODEL_DIR}"
echo "SUBMITTER = ${SUBMITTER}"
echo "CONTAINER_SUBMITTER_DIR = ${CONTAINER_SUBMITTER_DIR}"
echo "HOST_SUBMITTER_DOCKERFILE = ${HOST_SUBMITTER_DOCKERFILE}"
echo "CONFIG = ${CONFIG}"
echo "CONTAINER_MOUNTS = ${CONTAINER_MOUNTS}"
echo "TRAINING_SCRIPT_ARGS = ${TRAINING_SCRIPT_ARGS[*]}"
echo "CURRENT_DIR = ${CURRENT_DIR}"
echo "CONTAINER_ENVIRONMENT_VARIABLES_PATH = ${CONTAINER_ENVIRONMENT_VARIABLES_PATH}"
echo "RESULTS_DIR = ${RESULTS_DIR}"
echo "LOG_FILE_BASE = ${LOG_FILE_BASE}"
echo "SHM_SIZE = ${SHM_SIZE}"
echo "======================================"


# =================================================
# Training
# =================================================

# Build image
echo "================Build Docker image begin.............."

IMAGE_INFO=$(docker images ${DOCKER_IMAGE} |wc -l)

if [ ${IMAGE_INFO} -eq 2 ]; then
	echo "Docker image: ${DOCKER_IMAGE} already exists. Use this docker image. if you want to build new image, please delete this image first."
	echo "Command:./cleanup_image_container.sh  ${SUBMITTER} image"
	echo "or:     ./cleanup_image_container.sh  ${SUBMITTER} all #clean container and container at the same time"
else
	cd ${HOST_SUBMITTER_DIR}/docker_image
	bash ./gen_docker_image.sh ${DOCKER_IMAGE}
	if [ $? -ne 0 ]; then
		echo "Build docker image ${DOCKER_IMAGE} error."
		cd ${CURRENT_DIR}
		eixt 1
	fi
	cd ${CURRENT_DIR}

	echo "Built new docker image: ${DOCKER_IMAGE}"
fi

docker images ${DOCKER_IMAGE}
echo -e "================Build Docker image end.............\n"

#Setup docker container 
echo "================Setup Docker container begin.............."

CONTAINER_INFO=$(docker ps -a -f NAME=${CONTAINER_NAME} |wc -l)
if [ ${CONTAINER_INFO} -eq 2 ]; then
	echo "Docker container ${CONTAINER_NAME} already exists. Use this container. if you want to setup new container, please delete this image first."
	echo "Command:./cleanup_image_container.sh  ${SUBMITTER} container"
	echo "or:     ./cleanup_image_container.sh  ${SUBMITTER} all #clean container and container at the same time"

	CONTAINER_INFO=$(docker ps -f NAME=${CONTAINER_NAME} |wc -l)
	if [ ${CONTAINER_INFO} -eq 1 ]; then
		docker start ${CONTAINER_NAME}	
	fi
else
	docker run --rm --init --detach \
    		--net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    		--privileged=true \
    		--ulimit=stack=67108864 --ulimit=memlock=-1 \
    		-w ${MODEL_DIR} \
    		--shm-size="${SHM_SIZE}" \
    		--volume ${SOURCE_DATA_DIR}:${MAP_DATA_DIR} \
    		--volume ${PROJ_DIR}:${WORK_DIR} \
    		--name="${CONTAINER_NAME}" ${CONTAINER_MOUNTS} \
    		"${DOCKER_IMAGE}" sleep infinity

	# make sure container has time to finish initialization
	sleep 10
	docker exec -it "${CONTAINER_NAME}" true
	docker exec -it "${CONTAINER_NAME}" /bin/bash -c "source ${CONTAINER_ENVIRONMENT_VARIABLES_PATH};python3 prepare.py --name ${SUBMITTER} --data_dir ${MAP_DATA_DIR}"
fi

echo -e "================Setup Docker container end..............\n"


mkdir -p ${RESULTS_DIR}

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
        echo "source ${CONTAINER_ENVIRONMENT_VARIABLES_PATH};bash ./run_training.sh ${TRAINING_SCRIPT_ARGS[*]}"

        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
        fi

        # Run experiment
        docker exec -it "${CONTAINER_NAME}" /bin/bash -c "source ${CONTAINER_ENVIRONMENT_VARIABLES_PATH};bash ./run_training.sh ${TRAINING_SCRIPT_ARGS[*]}"
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done