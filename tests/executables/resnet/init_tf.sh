bash ../_utils/init_tf_cnn_benckmark.sh ../_utils

CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}/../..
DATA_DIR=${ROOT_DIR}/data/packages

# sys_name_str=`uname -a`
# if [[ "${sys_name_str}" =~ "aarch64" ]]; then
#     pip3 install ${DATA_DIR}/addons/tensorflow_addons*.whl
# fi

# pip3 install gin-config tensorflow_addons tensorflow_datasets tensorflow_model_optimization

pip3 install gin-config tensorflow_datasets tensorflow_model_optimization

pip3 uninstall -y protobuf
pip3 install "protobuf<4.0.0"  

python_version=$(python3 --version 2>&1 |awk '{print $2}'|awk -F '.' '{printf "%d.%d", $1,$2}')
if [ $python_version == 3.7 ]; then
    pip3 install numpy==1.21.6
else
    pip3 install numpy==1.23.3
fi

