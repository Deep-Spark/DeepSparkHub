#!/bin/bash

COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
${PYTHON_PATH} -m pip list | grep "^torch .*+corex" || {
  echo "ERROR: building mmcv requries the corex torch has been installed."
  exit 1
}

export MAX_JOBS=${MAX_JOBS}

${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit
# Return 0 status if all finished
exit 0

