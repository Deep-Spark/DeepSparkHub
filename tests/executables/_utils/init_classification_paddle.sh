#!/bin/bash

if [ -n "$1" ]; then
    _UTILS_DIR=$1
else
    _UTILS_DIR='../_utils'
fi

# Install packages
. $_UTILS_DIR/install_pip_pkgs.sh

pkgs=('scipy' 'scikit-learn==0.23.2' 'opencv-python' 'tqdm' "visualdl==2.3.0")


PY_VERSION=$(python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}')
if [ "$PY_VERSION" == "10" ];
then
   pkgs=('scipy' 'scikit-learn==1.1.0' 'opencv-python' 'tqdm' "visualdl==2.3.0")
   echo "$pkgs"
elif  [ "$PY_VERSION" == "11" ];
then
   pkgs=('scipy' 'scikit-learn==1.3.1' 'opencv-python' 'tqdm' "visualdl==2.3.0")
   echo "$pkgs"
else
   pkgs=('scipy' 'scikit-learn==0.24.0' 'opencv-python' 'tqdm' "visualdl==2.3.0")
   echo "$pkgs"
fi

install_pip_pkgs "${pkgs[@]}"
