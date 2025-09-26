#!/bin/bash
#This script is to check if needed package is installed, also download dataset and pre-trained weights if not exist.
. ../_utils/install_pip_pkgs.sh

CURRENT_MODEL_DIR=$(cd `dirname $0`; pwd)
PROJ_DIR="${CURRENT_MODEL_DIR}/../../"
PROJECT_DATA="${PROJ_DIR}/data/datasets"
MODEL_ZOO_DIR="${PROJ_DIR}/data/model_zoo"

pkgs=('requests' 'matplotlib' 'numpy' 'Pillow' 'scipy' 'tqdm' 'seaborn' 'pandas' 'thop' 'opencv-python' 'pycocotools' '--ignore-installed PyYAML')
install_pip_pkgs "${pkgs[@]}"

pip3 install tqdm==4.62.1

git clone https://gitee.com/deep-spark/deepsparkhub-gpl.git

cd ${PROJ_DIR}/deepsparkhub-gpl/cv/detection/yolov5-sample/pytorch

# Remove exist datas
if [[ -d "./datasets/coco128" ]]; then
    rm -rf ./datasets/coco128
fi

if [[ -d "./datasets/coco" ]]; then
    rm -rf ./datasets/coco
fi

if [[ -d "./weights" ]]; then
    rm -rf ./weights
fi
mkdir "weights"

if [[ -d "./datasets" ]]; then
    rm ./datasets
fi

# Build datas
if [[ ! -d "datasets" ]]; then
    echo "ln -s ${PROJECT_DATA} ./datasets"
    ln -s ${PROJECT_DATA} ./datasets
fi

if [[ ! -d "${PROJECT_DATA}/coco128" ]];then
    if [ -f "${PROJECT_DATA}/coco128.tgz" ]; then
        echo "Unarchive coco128.tgz"
        tar zxf "${PROJECT_DATA}/coco128.tgz" -C ./datasets/
    else
        echo "Error: Not found ${PROJECT_DATA}/coco128.tgz!"
    fi
else
    echo "Warning: coco128 exist!"
fi

if [[ -d "${PROJECT_DATA}/coco2017" ]];then
    if [[ -f "${PROJECT_DATA}/coco2017labels.zip" ]]; then
        echo "Unarchive coco2017labels.zip"
        unzip -q -d ./datasets/ "${PROJECT_DATA}/coco2017labels.zip"

        echo "ln -s ${PROJECT_DATA}/coco2017/train2017 ./datasets/coco/images/"
        ln -s ${PROJECT_DATA}/coco2017/train2017 ./datasets/coco/images/

        echo "ln -s ${PROJECT_DATA}/coco2017/val2017 ./datasets/coco/images/"
        ln -s ${PROJECT_DATA}/coco2017/val2017 ./datasets/coco/images/
    else
        echo "Error: Not found ${PROJECT_DATA}/coco2017labels.zip!"
    fi
else
    echo "Warning: Not found coco2017!"
fi

if [[ -f "${MODEL_ZOO_DIR}/yolov5s.pt" ]];then
    echo "ln -s ${MODEL_ZOO_DIR}/yolov5s.pt ./weights"
    ln -s ${MODEL_ZOO_DIR}/yolov5s.pt ./weights
fi

if [[ -f "/opt/rh/gcc-toolset-11/enable" ]];then
    source /opt/rh/gcc-toolset-11/enable
fi