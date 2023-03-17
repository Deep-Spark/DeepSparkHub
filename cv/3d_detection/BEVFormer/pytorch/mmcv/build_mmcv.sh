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
${PYTHON_PATH} -m pip list | grep "^torch .*+corex" || {
  echo "ERROR: building mmcv requries the corex torch has been installed."
  exit 1
}

export MAX_JOBS=${MAX_JOBS}

FORCE_CUDA=1 MMCV_WITH_OPS=1 ${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export MMCV_LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"
FORCE_CUDA=1 MMCV_WITH_OPS=1 ${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0
