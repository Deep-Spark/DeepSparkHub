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
