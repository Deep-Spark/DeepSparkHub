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


#!/bin/bash

PYTHON_PATH=$(which python3)

echo "build firefly"
COREX_VERSION=${COREX_VERSION:-latest}
if [[ "${COREX_VERSION}" == "latest" || -z "${COREX_VERSION}" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
FIREFLY_VERSION_IDENTIFIER="corex.${COREX_VERSION}"
export FIREFLY_VERSION_IDENTIFIER=${FIREFLY_VERSION_IDENTIFIER}

${PYTHON_PATH} setup.py build
${PYTHON_PATH} setup.py bdist_wheel

PKG_DIR="./dist"
rm -rf build_pip
if [[ ! -d "build_pip" ]]; then
  mkdir build_pip
fi
pip_pkg="$(ls -t ${PKG_DIR} | grep "firefly" | head -1)"
cp ${PKG_DIR}/${pip_pkg} build_pip

exit 0