#!/bin/bash
EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}


: ${CXX:="g++"}
export CXX

source $(cd `dirname $0`; pwd)/../_utils/which_install_tool.sh

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

if command_exists apt; then
	$prefix_sudo apt install -y git numactl
elif command_exists dnf; then
	$prefix_sudo dnf install -y git numactl
else
	$prefix_sudo yum install -y git numactl
fi
sys_name_str=`uname -a`
if [[ "${sys_name_str}" =~ "aarch64" ]]; then
    pip3 install "git+https://github.com/mlperf/logging.git@1.0-branch" pybind11 ujson
else
    pip3 install "git+https://github.com/mlperf/logging.git@1.0-branch" pybind11==2.9.2 ujson==1.35
fi

pip3 list | grep -w "wheel" || pip3 install wheel
# pip3 list | grep -w "numpy" | grep -w "1.23.5" || pip3 install numpy==1.23.5
pip3 install numpy>=1.26.4
pip3 install cython
# pip3 install "git+https://github.com/NVIDIA/cocoapi.git@v0.6.0#subdirectory=PythonAPI"
pip3 install pycocotools==2.0.8

CUR_PATH=$(cd `dirname $0`; pwd)
DATA_PATH=$CUR_PATH/../../data/datasets/coco2017/

if [[ "$(uname -m)" == "aarch64" ]]; then
    source /opt/rh/gcc-toolset-11/enable
fi

cd ../../cv/detection/ssd/pytorch/ && bash ./clean_ssd.sh && bash ./build_ssd.sh && bash ./install_ssd.sh "$@";  check_status
DATA_PATH_BBOX=../../../..

python3 prepare-json.py --keep-keys ${DATA_PATH}/annotations/instances_val2017.json ${DATA_PATH_BBOX}/bbox_only_instances_val2017.json "$@";  check_status
python3 prepare-json.py ${DATA_PATH}/annotations/instances_train2017.json ${DATA_PATH_BBOX}/bbox_only_instances_train2017.json "$@";  check_status


cd - 
#echo "init finished!"
exit ${EXIT_STATUS}
