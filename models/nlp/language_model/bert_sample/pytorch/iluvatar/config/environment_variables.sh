# /***************************************************************************************************
# * Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# * Copyright Declaration: This software, including all of its code and documentation,
# * except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# * Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# * Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# * CoreX. No user of this software shall have any right, ownership or interest in this software and
# * any use of this software shall be in compliance with the terms and conditions of the End User
# * License Agreement.
#  **************************************************************************************************/

# =================================================
# Constants
# =================================================

CURRENT_DIR=$(cd `dirname $0`; pwd)  # /path/to/proj/benchmarks/${MODEL}/pytorch
PROJ_DIR="${CURRENT_DIR}/../../.."
SUBMMIT_DIR="${PROJ_DIR}/iluvatar/${MODEL}"
SDK_DIR="${SUBMMIT_DIR}/sdk_installers"
if [ ! -d "${SDK_DIR}" ]; then
    echo "WARN: Not found ${SDK_DIR}, set SDK_DIR to ${PROJ_DIR}/iluvatar/sdk_installers"
    SDK_DIR="${PROJ_DIR}/iluvatar/sdk_installers"
fi
SDK_BAK_DIR="${SDK_DIR}.bak"

DRIVER_FILE_PATH=""
CUDA_FILE_PATH=""


# =================================================
# Check environment
# =================================================

if [ -d "${SDK_DIR}" ]; then
    search_cuda_results=`find ${SDK_DIR} -name "*cuda*.run"`
    for cuda_file_name in $search_cuda_results; do
        CUDA_FILE_PATH="${cuda_file_name}"
    done
fi


if [ -d "/usr/local/cuda" ]; then
    # Found cuda

    # Mapping host cuda to container
    cuda_dirs=`find /usr/local -maxdepth 1 -name "cuda*"`
    for cuda_dir in $cuda_dirs; do
        CONTAINER_MOUNTS="$CONTAINER_MOUNTS -v ${cuda_dir}:${cuda_dir}"
    done

    # Blocking install cuda
    mkdir -p "${SDK_BAK_DIR}"
    if [ -n "${CUDA_FILE_PATH}" ] && [ -f "${CUDA_FILE_PATH}" ]; then
        echo "WARN: Move ${CUDA_FILE_PATH} to ${SDK_BAK_DIR}"
        mv "${CUDA_FILE_PATH}" "${SDK_BAK_DIR}"
    fi
fi


# =================================================
# Export variables
# =================================================

export CONTAINER_MOUNTS="${CONTAINER_MOUNTS} -v /dev:/dev -v /usr/src/:/usr/src -v /lib/modules/:/lib/modules --cap-add=ALL"
export SDK_ARGUMENTS="cuda_=-- --silent --toolkit;corex-installer=-- --silent --cudapath=/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/corex/lib64"
SYS_ENVS="/root/miniconda/bin:/root/miniconda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="/usr/local/corex/bin:${SYS_ENVS}"
export CONTAINER_MOUNTS
