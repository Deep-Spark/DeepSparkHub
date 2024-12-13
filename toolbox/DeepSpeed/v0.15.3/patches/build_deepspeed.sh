#!/bin/bash

COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
PLATFORM_ID=$(uname -i)
${PYTHON_PATH} -c "import torch;print(torch.__version__)" || {
  echo "ERROR: building vision requries torch has been installed."
  exit 1
}
PY_VERSION=`${PYTHON_PATH} -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
OS_ID=$(awk -F= '/^ID=/{print $2}' /etc/os-release | tr -d '"')
if [[ "${OS_ID}" == "ubuntu" ]]; then
  sudo apt-get install libaio-dev -y || exit
elif [[ "${OS_ID}" == "centos" ]]; then
  sudo yum install libaio libaio-devel -y || exit
else
  echo "Warning: unable to identify OS ..."
fi

pip3 install -r requirements/requirements-bi.txt

# ${PYTHON_PATH} -m pip install -r requirements_dev.txt || exit

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export DEEPSPEED_LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"

export MAX_JOBS=${MAX_JOBS}

# export DS_BUILD_OPS=0
arch=$(uname -m)

if [ "$arch" == "aarch64" ]; then
    echo "This is an ARM architecture"
    export DS_BUILD_CPU_ADAM=0
elif [ "$arch" == "x86_64" ]; then
    echo "This is an x86 architecture"
    export DS_BUILD_CPU_ADAM=1
else
    echo "Unknown architecture: $arch"
fi
export DS_BUILD_CPU_LION=1
export DS_BUILD_FUSED_LION=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_FUSED_LAMB=1
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=1
export DS_BUILD_QUANTIZER=1
export DS_BUILD_CPU_ADAGRAD=1
export DS_BUILD_RANDOM_LTD=1
export DS_BUILD_SPATIAL_INFERENCE=1
export DS_BUILD_TRANSFORMER_INFERENCE=1
export DS_BUILD_STOCHASTIC_TRANSFORMER=1
export DS_BUILD_UTILS=1
export DS_ACCELERATOR=cuda
export DS_BUILD_AIO=1

export DS_BUILD_EVOFORMER_ATTN=0
export DS_BUILD_SWIGLU=1
export DS_BUILD_FUSED_ROPE=1
export DS_BUILD_FUSED_LAYERNORM=1
export DS_BUILD_GDS=1


${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0
