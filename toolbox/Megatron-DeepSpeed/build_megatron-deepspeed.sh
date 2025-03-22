#!/bin/bash

PYTHON_PATH=$(which python3)

echo "build megatronspeed"
COREX_VERSION=${COREX_VERSION:-latest}
if [[ "${COREX_VERSION}" == "latest" || -z "${COREX_VERSION}" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
MEGATRONSPEED_VERSION_IDENTIFIER="corex.${COREX_VERSION}"
export MEGATRONSPEED_VERSION_IDENTIFIER=${MEGATRONSPEED_VERSION_IDENTIFIER}

${PYTHON_PATH} setup.py build
${PYTHON_PATH} setup.py bdist_wheel

PKG_DIR="./dist"
rm -rf build_pip
if [[ ! -d "build_pip" ]]; then
  mkdir build_pip
fi

pip_pkg="$(ls -t ${PKG_DIR} | grep "megatron" | head -1)"
cp ${PKG_DIR}/${pip_pkg} build_pip

exit 0