CURRENT_DIR=$(cd `dirname $0`; pwd)

source ../_utils/global_environment_variables.sh

unset http_proxy
unset https_proxy

ROOT_DIR=${CURRENT_DIR}/../..
export DATA_DIR=${ROOT_DIR}/data/datasets

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd ${ROOT_DIR}/official/cv/classification/resnet/tensorflow
ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 0.01 --run_script \
bash run_train_resnet50_distributed_imagenette.sh "$@";  check_status

exit ${EXIT_STATUS}