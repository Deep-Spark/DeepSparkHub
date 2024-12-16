#!/bin/bash

COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
${PYTHON_PATH} -c "import torch;print(torch.__version__)" || {
  echo "ERROR: building vision requries torch has been installed."
  exit 1
}
PY_VERSION=`${PYTHON_PATH} -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
OS_ID=$(awk -F= '/^ID=/{print $2}' /etc/os-release | tr -d '"')

pip3 install -r requirements/requirements.txt

# ${PYTHON_PATH} -m pip install -r requirements_dev.txt || exit

if [[ "${COREX_VERSION}" == "latest" || -z "${COREX_VERSION}" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export COLOSSALAI_LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"

export MAX_JOBS=${MAX_JOBS}

${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0
