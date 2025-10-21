source ../_utils/global_environment_variables.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ../_utils/get_num_devices.sh

# export LD_PRELOAD=/usr/local/lib/libmpi.so:${LD_PRELOAD}

: ${BATCH_SIZE:=6}

sys_name_str=`uname -a`
if [[ "${sys_name_str}" =~ "aarch64" ]]; then
    export TF_FORCE_SINGLE_THREAD=1
fi

# if [ "${CI}" == "true" ]; then
#     if [ ! -d "/usr/local/lib/openmpi" ]; then
#         echo "Not found /usr/local/lib/openmpi, Installing mpi ......"
#         install-mpi
#     fi
#     export HOROVOD_RUN_ARGS=" "
#     export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/openmpi:${LD_LIBRARY_PATH}
# fi

set -euox pipefail

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
ROOT_DIR=${current_path}"/../../"
SRC_DIR=${ROOT_DIR}../models/nlp/language_model/bert/tensorflow/base
DATA_DIR=${ROOT_DIR}data/datasets
MODEL_DIR=${ROOT_DIR}data/model_zoo

nonstrict_mode_args=""
if [ "${RUN_MODE}" != "strict" ]; then
    nonstrict_mode_args="--stop_threshold 0.6"
fi

cd $SRC_DIR
bash init.sh
bash run_multi_card_FPS.sh \
            --input_files_dir=${DATA_DIR}/bert_pretrain_tf_records/train_data \
            --init_checkpoint=${MODEL_DIR}/bert_pretrain_tf_ckpt/model.ckpt-28252 \
            --eval_files_dir=${DATA_DIR}/bert_pretrain_tf_records/eval_data \
            --train_batch_size=${BATCH_SIZE} \
            --bert_config_file=${MODEL_DIR}/bert_pretrain_tf_ckpt/bert_config.json \
            --display_loss_steps=10 ${nonstrict_mode_args} \
            "$@"

exit $?
