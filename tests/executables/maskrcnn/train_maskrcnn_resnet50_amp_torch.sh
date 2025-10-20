source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=1}

export PYTORCH_DISABLE_VEC_KERNEL=1
export PT_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1

OUTPUT_DIR=${PROJECT_DIR}/output/maskrcnn/$0
if [[ -d ${OUTPUT_DIR} ]]; then
    mkdir -p ${OUTPUT_DIR}
fi


ixdltest-check --nonstrict_mode_args="--epoch ${NONSTRICT_EPOCH}" -b 10 --run_script \
python3 ${PROJECT_DIR}../cv/detection/maskrcnn/pytorch/train.py \
--model maskrcnn_resnet50_fpn \
--data-path ${PROJECT_DIR}/data/datasets/VOC2012_sample \
--amp \
--lr 0.001 \
--batch-size ${BATCH_SIZE} \
--output-dir ${OUTPUT_DIR} \
"$@"; check_status

rm -fr ${OUTPUT_DIR}
exit ${EXIT_STATUS}
