source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=8}

ROOT_DIR="$(cd "$(dirname "$0")/../.."; pwd)"
SRC_DIR=$ROOT_DIR/../models/audio/speech_recognition/conformer/pytorch
DATA_DIR=$ROOT_DIR/data
export DRT_MEMCPYUSEKERNEL=20000000000

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

source ../_utils/fix_import_sklearn_error_libgomp_d22c30c5.sh

cd $SRC_DIR
bash run_training.sh --data_dir=$DATA_DIR \
    --max_steps=800 \
    --quality_target=1.6 \
    --batch_size=${BATCH_SIZE} \
    --eval_freq=400 \
    --ddp \
    --max_steps=800 \
    --quality_target=1.6 \
    --eval_freq=400; check_status

exit ${EXIT_STATUS}