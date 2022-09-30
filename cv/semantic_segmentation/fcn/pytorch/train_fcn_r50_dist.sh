CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}/../../torchvision/pytorch

python3 -m torch.distributed.launch --nproc_per_node="auto" --use_env \
${ROOT_DIR}/train.py --model fcn_resnet50 "$@"