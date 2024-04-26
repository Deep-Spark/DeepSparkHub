#!/bin/bash

COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
${PYTHON_PATH} -m pip list | grep "^torch .*+corex" || {
  echo "ERROR: building mmcv requries the corex torch has been installed."
  exit 1
}

export MAX_JOBS=${MAX_JOBS}

PY_VERSION=$(python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}')

if [ "$PY_VERSION" == "6" ] || [ "$PY_VERSION" == "7" ];
then
   export CPATH=$CPATH:/usr/local/include/python3.${PY_VERSION}m/
else
   export CPATH=$CPATH:/usr/local/include/python3.${PY_VERSION}/
fi

SIGN=${SIGN:-$(ixsmi -L)}
if [[ $SIGN =~ MR || $SIGN =~ BI-V150 ]]
then
    export CMAKE_CUDA_ARCHITECTURES=ivcore11
else
    export CMAKE_CUDA_ARCHITECTURES=ivcore10
fi

${PYTHON_PATH} setup.py bdist_wheel -d build_pip 2>&1 | tee compile.log || exit
# Return 0 status if all finished
exit 0

