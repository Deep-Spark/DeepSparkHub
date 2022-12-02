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

MODEL="ssd"
export MODEL
DOCKER_IMAGE="model_zoo:${MODEL}"
NEXP=1

# TODO: Add to Dockerfile
WORK_DIR="/workspace/model_zoo/ssd/pytorch"
MODEL_DIR="${WORK_DIR}/base"

CURRENT_DIR=$(cd `dirname $0`; pwd)
PROJ_DIR="${CURRENT_DIR}/../"
BUILD_EXTENSION_DIR="${CURRENT_DIR}/build"
BUILD_EXTENSION_PACKAGE_NAME="ext_ops"

BASE_DOCKERFILE_PATH="${CURRENT_DIR}/BaseDockerfile"
HOST_DOCKERFILE_PATH="${CURRENT_DIR}/Dockerfile"

SOURCE_DATA_DIR=""
MAP_DATA_DIR="/mnt/dataset/model_zoo/${MODEL}"
MAP_BACKBONE_DIR="/mnt/dataset/model_zoo/${MODEL}/resnet34-333f7ec4.pth"
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
            TRAINING_SCRIPT_ARGS=${TRAINING_SCRIPT_ARGS/$arg/"--data_dir ${MAP_DATA_DIR} --backbone_path ${MAP_BACKBONE_DIR}"}
        else
            SOURCE_DATA_DIR=${!i}
            TRAINING_SCRIPT_ARGS=${TRAINING_SCRIPT_ARGS/"--data_dir ${!i}"/"--data_dir ${MAP_DATA_DIR} --backbone_path ${MAP_BACKBONE_DIR}"}
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
CONTAINER_NAME="model_zoo-${MODEL}-${SUBMITTER}-container"

if [ ! -f "${HOST_ENVIRONMENT_VARIABLES_PATH}" ]; then
    touch "${HOST_ENVIRONMENT_VARIABLES_PATH}"
fi

source ${HOST_ENVIRONMENT_VARIABLES_PATH}

RESULTS_DIR="${PROJ_DIR}/${SUBMITTER}/${MODEL}/results"
LOG_FILE_BASE="${RESULTS_DIR}/config_${CONFIG}_experiment"

echo "======================================"
echo "Arguments"
echo "---------"

echo "CLEAR_CACHES = ${CLEAR_CACHES}"
echo "CLEAR_CONTAINER = ${CLEAR_CNT}"
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

# Cleanup container
cleanup_docker() {
    docker container rm -f "${CONTAINER_NAME}" || true
}
if [ "${CLEAR_CNT}" -eq 1 ]; then
    cleanup_docker
    trap 'set -eux; cleanup_docker' EXIT
fi


container_status=`docker ps | grep ${CONTAINER_NAME}`
if [[ ! ${container_status} =~ ${CONTAINER_NAME} ]]; then
  # Clean built extension
  if [ -d "${BUILD_EXTENSION_DIR}" ]; then
      echo "WARN: Delete built extension"
      rm -rf "${BUILD_EXTENSION_DIR}"
      rm -rf ${CURRENT_DIR}/${BUILD_EXTENSION_PACKAGE_NAME}.*.so
      echo "extension file: "${CURRENT_DIR}/${BUILD_EXTENSION_PACKAGE_NAME}.*.so""
  fi


  # Build image
  if [ -f "${HOST_DOCKERFILE_PATH}" ]; then
      echo "WARN: Remove previous Dockerfile"
      rm -f "${HOST_DOCKERFILE_PATH}"
  fi

  echo "WARN: cp BaseDockerfile to Dockerfile"
  cp "${BASE_DOCKERFILE_PATH}" "${HOST_DOCKERFILE_PATH}"

  if [ ${SUBMITTER} = "nvidia" ]; then
      echo "Nvidia Dockerfile build from Nvidia NGC Images."
      cat "${HOST_SUBMITTER_DOCKERFILE}" > "${HOST_DOCKERFILE_PATH}"
  elif [ -f "${HOST_SUBMITTER_DOCKERFILE}" ]; then
      echo "WARN: Found submitter's Dockerfile, merging submitter's Dockerfile to Dockerfile"
      cat "${HOST_SUBMITTER_DOCKERFILE}" >> "${HOST_DOCKERFILE_PATH}"
  fi

  docker build -t ${DOCKER_IMAGE} ./

  # Setup container by Dockerfile
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
  # TODO: Uncomment
  sleep 5
  docker exec -it "${CONTAINER_NAME}" true

  mkdir -p ${RESULTS_DIR}
  docker exec -it "${CONTAINER_NAME}" sh -c "chmod 777 run_training.sh"

  # TODO: Remove pip source
  docker exec -it "${CONTAINER_NAME}" /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"

  docker exec -it "${CONTAINER_NAME}" /bin/bash -c "source ${CONTAINER_ENVIRONMENT_VARIABLES_PATH};python3 prepare.py --name ${SUBMITTER} --data_dir ${MAP_DATA_DIR}"
fi

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