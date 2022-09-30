#!/bin/bash

# =================================================
# Constants
# =================================================

CURRENT_DIR=$(pwd)
BASE_DIR=$(cd `dirname $0`; pwd)

if [ $# -lt 2 ];then
	echo -e "\nAbsent argument. Usage: $0  <SUBMITTER> image|container|all\n"
	exit 1;
fi

MODEL="cpm"
SUBMITTER=$1

DOCKER_IMAGE="perf-${SUBMITTER}:${MODEL}_t"
CONTAINER_NAME="perf-${SUBMITTER}-${MODEL}_t-container"

BUILD_EXTENSION_DIR="${BASE_DIR}/build"
BUILD_EXTENSION_PACKAGE_NAME="ext_ops"

if [ $2 == "container" -o $2 == "all" ]; then 
	# Clean built extension intermidiate files
	if [ -d "${BUILD_EXTENSION_DIR}" ]; then
    		echo "Delete built extension"
    		rm -rf "${BUILD_EXTENSION_DIR}"
    		rm -rf ${BASE_DIR}/${BUILD_EXTENSION_PACKAGE_NAME}.*.so
    		echo "extension file: "${BASE_DIR}/${BUILD_EXTENSION_PACKAGE_NAME}.*.so""
	fi

	docker rm -f "${CONTAINER_NAME}"
fi

if [ $2 == "image" -o $2 == "all" ]; then 
	docker rmi -f "${DOCKER_IMAGE}"
fi

cd ${CURRENT_DIR}
