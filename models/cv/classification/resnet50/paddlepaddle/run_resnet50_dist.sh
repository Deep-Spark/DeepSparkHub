#!/bin/bash

export PYTHONPATH=./:${PYTHONPATH}

pip3 install -r requirements.txt

python3 -m paddle.distributed.launch -ips=127.0.0.1 train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml
exit $?