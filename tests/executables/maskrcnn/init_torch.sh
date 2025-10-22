bash ../_utils/init_detection_torch.sh ../_utils

CURRENT_MODEL_DIR=$(cd `dirname $0`; pwd)
PROJ_DIR="${CURRENT_MODEL_DIR}/../../"
PROJECT_DATA="${PROJ_DIR}/data/datasets"

if [[ ! -d "${PROJECT_DATA}/VOC2012_sample" ]]; then
    tar zxf ${PROJECT_DATA}/VOC2012_sample.tgz -C ${PROJECT_DATA}
fi

cd ${PROJ_DIR}/models/cv/detection/maskrcnn/pytorch/
OSNAME=$(cat /proc/version)
# install the requirement
if [[ "${OSNAME}" == *"aarch64"* ]] 
then
	pip3 install -r requirements_aarch64.txt
fi