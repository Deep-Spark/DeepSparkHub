#!/bin/bash

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
