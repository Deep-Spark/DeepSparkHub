CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}

python3 -m torch.distributed.launch --nproc_per_node="auto" --use_env \
${ROOT_DIR}/run_train.py --model dunet_resnet50 "$@"