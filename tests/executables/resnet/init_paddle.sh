#!/bin/bash
bash ../_utils/init_classification_paddle.sh ../_utils

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip3 install protobuf==3.20.3
pip3 install pyyaml

CUR_DIR=$(cd `dirname $0`; pwd)
PRJ_DIR=${CUR_DIR}/../..
DATASET_DIR=${PRJ_DIR}/data/datasets

if [ ! -d "${DATASET_DIR}/flowers102" ]; then
    tar zxf ${DATASET_DIR}/flowers102.tgz -C ${DATASET_DIR}
fi

RESNET_PADDLE_DIR=${PRJ_DIR}/official/cv/classification/resnet/paddle
cd ${RESNET_PADDLE_DIR}
pip3 install -r requirements.txt

a=$(pip3 show paddlepaddle|awk '/Version:/ {print $NF}'); b=(${a//+/ }); c=(${b//./ })
if [[ ${c[0]} -eq 2 && ${c[1]} -le 5 ]]; then
  rm -rf ppcls && ln -s ppcls_2.5 ppcls
  mkdir -p data/datasets
  ln -s ${DATASET_DIR}/flowers102 ${RESNET_PADDLE_DIR}/data/datasets/flowers102
else
  rm -rf ppcls && ln -s ppcls_2.6 ppcls
  mkdir -p dataset
  ln -s ${DATASET_DIR}/flowers102 ${RESNET_PADDLE_DIR}/dataset/flowers102
fi
