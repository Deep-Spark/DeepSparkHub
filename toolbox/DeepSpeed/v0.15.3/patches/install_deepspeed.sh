#!/bin/bash

TARGET_DIR=${TARGET_DIR:-}

PYTHON_PATH=$(which python3)
PYTHON_DIST_PATH=${TARGET_DIR}/lib/python3/dist-packages

PKG_DIR="build_pip"
PKG_NAME="deepspeed"

if [[ ! -d ${PKG_DIR} ]]; then
  echo "ERROR: Package directory ${PKG_DIR} doesn't exist"
  exit 1
fi

latest_pkg="$(ls -t ${PKG_DIR} | grep ${PKG_NAME} | head -1)"
if [[ "${latest_pkg}" == "" ]]; then
  echo "ERROR: Cannot find latest ${PKG_NAME} package"
  exit 1
else
  echo "INFO: Found latest package ${latest_pkg} in directory ${PKG_DIR}"
fi

${PYTHON_PATH} -m pip uninstall ${PKG_NAME} -y
${PYTHON_PATH} -m pip install ${PKG_DIR}/${latest_pkg} || exit

# Return 0 status if all finished
exit 0
