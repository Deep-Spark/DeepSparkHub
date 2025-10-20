#!/bin/bash

CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}/../..
ROOT_DIR=${CURRENT_DIR}/../..
DATA_DIR=${ROOT_DIR}/data/datasets/imagenette_tfrecord

if [ -n "$1" ]; then
    _UTILS_DIR=$1
else
    _UTILS_DIR='../_utils'
fi

# Install packages
. $_UTILS_DIR/install_pip_pkgs.sh

pkgs=('absl-py')

install_pip_pkgs "${pkgs[@]}"
pip3 install ${_UTILS_DIR}/../../data/packages/DLLogger-1.0.0-py3-none-any.whl


if [ ! -d "${DATA_DIR}" ]; then
    cd ${ROOT_DIR}/data/datasets/
    tar -xzvf imagenette_tfrecord.tgz
fi
