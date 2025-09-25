# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


PIPCMD=pip3

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

## Install packages
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    $prefix_sudo apt update
    sudo_path='command -v sudo'
    if [ -z "${sudo_path}" ]; then
        echo "Install sudo"
        $prefix_sudo apt install -y sudo
    fi
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        $prefix_sudo apt install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        $prefix_sudo apt install -y unzip
    fi
    $prefix_sudo apt -y install libgl1-mesa-glx
    pyver=`python3 -c 'import sys; print(sys.version_info[:][0])'`
    pysubver=`python3 -c 'import sys; print(sys.version_info[:][1])'`
    $prefix_sudo apt -y install python${pyver}.${pysubver}-dev
elif [[ ${ID} == "centos" ]]; then
    sudo_path='command -v sudo'
    if [ -z "${sudo_path}" ]; then
        echo "Install sudo"
        $prefix_sudo yum install -y sudo
    fi
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        $prefix_sudo yum install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        $prefix_sudo yum install -y unzip
    fi
    $prefix_sudo yum -y install mesa-libGL
else
    sudo_path='command -v sudo'
    if [ -z "${sudo_path}" ]; then
        echo "Install sudo"
        $prefix_sudo yum install -y sudo
    fi
    cmake_path=`command -v cmake`
    if [ -z "${cmake_path}" ]; then
        echo "Install cmake"
        $prefix_sudo yum install -y cmake
    fi
    unzip_path=`command -v unzip`
    if [ -z "${unzip_path}" ]; then
        echo "Install unzip"
        $prefix_sudo yum install -y unzip
    fi
    $prefix_sudo yum -y install mesa-libGL
fi

# Fix No module named 'urllib3.packages.six'
sys_name_str=`uname -a`
if [[ "${sys_name_str}" =~ "aarch64" ]]; then
    pip3 install urllib3 requests --upgrade
fi
