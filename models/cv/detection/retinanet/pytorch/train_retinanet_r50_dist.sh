CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}

python3 -m torch.distributed.launch --nproc_per_node="auto" --use_env \
${ROOT_DIR}/train.py --model maskrcnn_resnet50_fpn --wd 0.000001 --lr 0.01 "$@"