#!/bin/bash

if [ -n "$1" ]; then
    _UTILS_DIR=$1
else
    _UTILS_DIR='../_utils'
fi

# Install packages
. $_UTILS_DIR/install_pip_pkgs.sh

pkgs=('scipy==1.12.0' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm')

install_pip_pkgs "${pkgs[@]}"