# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


# check current directory
: "${PROJ_DIR:=$(cd `dirname $0`; pwd)}"
if [ ! -d "${PROJ_DIR}/executables" ]; then
    echo "CurrentDirectory = ${PROJ_DIR}"
    echo "ERROR: Current directory is not found executables directory, exit 1."
    echo "Please set <PROJ_DIR> variable to deeplearningsamples, PROJ_DIR=<DIR> bash quick_build_environment.sh"
    exit 1
fi

cd ${PROJ_DIR}

echo "Current directory: `pwd`"

bash ./prepare_dataset.sh
bash ./prepare_system_environment.sh
bash ./prepare_python_environment.sh