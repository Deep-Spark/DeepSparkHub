# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


PIPCMD=pip3

PROJ_DIR=$(cd `dirname $0`; pwd)
PACKAGES_DIR="${PROJ_DIR}/data/packages"

if [ -d "${PACKAGES_DIR}" ]; then
    #$PIPCMD install --no-index --find-links=./packages numpy==1.19.5
    $PIPCMD install --no-index --find-links=${PACKAGES_DIR} scikit-build
else
    $PIPCMD install scikit-build
    #$PIPCMD install numpy==1.19.5
fi


# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

cd "${PROJ_DIR}/tools/dltest"
# $prefix_sudo python3 setup.py install
$prefix_sudo $PIPCMD install .