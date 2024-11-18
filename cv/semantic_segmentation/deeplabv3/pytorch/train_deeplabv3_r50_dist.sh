CURRENT_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR=${CURRENT_DIR}

python3 -m torch.distributed.launch --nproc_per_node="auto" --use_env \
${ROOT_DIR}/train.py --model deeplabv3_resnet50 --batch-size 8 --wd 0.000001 "$@"