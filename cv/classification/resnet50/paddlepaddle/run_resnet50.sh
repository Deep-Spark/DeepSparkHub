#!/bin/bash

export PYTHONPATH=./:${PYTHONPATH}

pip3 install -r requirements.txt

python3 train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml
exit $?