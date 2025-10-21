#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

clang_version=`clang --version | grep "clang version 16."`
if [[ "${clang_version}" != "" ]]; then
  echo "Not support LLVM16 now!"
  exit 0
fi

COREX_VERSION=${COREX_VERSION:-latest}

PYTHON_PATH=$(which python3)

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export NUMBA_LOCAL_IDENTIFIER="corex.${COREX_VERSION}"

${PYTHON_PATH} setup.py bdist_wheel -d build_pip 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

# Return 0 status if all finished
exit 0
