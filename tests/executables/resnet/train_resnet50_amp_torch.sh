
source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=256}

OUTPUT_DIR=${PROJECT_DIR}/output/resnet/$0
if [[ -d ${OUTPUT_DIR} ]]; then
    mkdir -p ${OUTPUT_DIR}
fi

ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 8 --run_script \
python3 ${PROJECT_DIR}/cv/classification/resnet50/pytorch/train.py \
--data-path ${PROJECT_DIR}/data/datasets/imagenette \
--batch-size ${BATCH_SIZE} \
--lr 0.01 \
--amp \
--output-dir ${OUTPUT_DIR} \
"$@" ;check_status

rm -fr ${OUTPUT_DIR}
exit ${EXIT_STATUS}
