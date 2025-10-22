#!/bin/bash
source ../_utils/global_environment_variables.sh
source ../_utils/set_paddle_environment_variables.sh
source ../_utils/get_num_devices.sh

OUTPUT_DIR=${PROJECT_DIR}/output/resnet/$0
if [[ -d ${OUTPUT_DIR} ]]; then
    mkdir -p ${OUTPUT_DIR}
fi

RESNET_PADDLE_DIR=${PROJECT_DIR}../models/cv/classification/resnet50/paddlepaddle/
cd ${RESNET_PADDLE_DIR}

ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 8 --run_script \
bash run_resnet50_dist.sh \
"$@" ;check_status

rm -fr ${OUTPUT_DIR}
exit ${EXIT_STATUS}