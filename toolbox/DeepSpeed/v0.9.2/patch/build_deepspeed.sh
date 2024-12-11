#!/bin/bash
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
${PYTHON_PATH} -c "import torch;print(torch.__version__)" || {
  echo "ERROR: building vision requries torch has been installed."
  exit 1
}

# OS_ID=$(awk -F= '/^ID=/{print $2}' /etc/os-release | tr -d '"')
# if [[ "${OS_ID}" == "ubuntu" ]]; then
#   sudo apt-get install libaio-dev -y || exit
# elif [[ "${OS_ID}" == "centos" ]]; then
#   sudo yum install libaio libaio-devel -y || exit
# else
#   echo "Warning: unable to identify OS ..."
# fi

pip3 install -r requirements/requirements-bi.txt

# ${PYTHON_PATH} -m pip install -r requirements_dev.txt || exit

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export DEEPSPEED_LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}.${OS_ID}"

export MAX_JOBS=${MAX_JOBS}

export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_SPARSE_ATTN=1
export DS_BUILD_TRANSFORMER=1
export DS_BUILD_CPU_ADAGRAD=1
export DS_BUILD_RANDOM_LTD=0
export DS_BUILD_TRANSFORMER_INFERENCE=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=1
export DS_BUILD_UTILS=1
export DS_BUILD_AIO=1

${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0
