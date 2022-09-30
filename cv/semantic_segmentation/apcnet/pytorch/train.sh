CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:2}
