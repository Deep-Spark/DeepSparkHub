#!/bin/bash

# /***************************************************************************************************
# * Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# * Copyright Declaration: This software, including all of its code and documentation,
# * except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# * Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# * Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# * CoreX. No user of this software shall have any right, ownership or interest in this software and
# * any use of this software shall be in compliance with the terms and conditions of the End User
# * License Agreement.
#  **************************************************************************************************/

CURRENT_DIR=$(cd `dirname $0`; pwd)

set -euox pipefail

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

source $(cd `dirname $0`; pwd)/../_utils/which_install_tool.sh
if command_exists apt; then
	$prefix_sudo apt install -y git numactl
elif command_exists dnf; then
	$prefix_sudo dnf install -y git numactl
else
	$prefix_sudo yum install -y git numactl
fi

if [ "$(ulimit -n)" -lt "1048576" ]; then
	ulimit -n 1048576
fi

# prepare data
cd  ${CURRENT_DIR}/../../data/datasets/bert_mini

if [[ ! -d "${CURRENT_DIR}/../../data/datasets/bert_mini/2048_shards_uncompressed" ]]; then
    echo "Unarchive 2048_shards_uncompressed_mini"
    tar -zxf 2048_shards_uncompressed_mini.tar.gz
fi
if [[ ! -d "${CURRENT_DIR}/../../data/datasets/bert_mini/eval_set_uncompressed" ]]; then
    echo "Unarchive eval_set_uncompressed.tar.gz"
    tar -zxf eval_set_uncompressed.tar.gz
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
    set +euox pipefail
    source /opt/rh/gcc-toolset-11/enable
    set -euox pipefail
fi


# install sdk
cd ${CURRENT_DIR}/../../nlp/language_model/bert_sample/pytorch/base
pip3 install -r requirements.txt
$prefix_sudo python3 setup.py install



if [ "$?" != "0" ]; then
    echo "init torch : failed."
    exit 1
fi

echo "init torch : completed."
