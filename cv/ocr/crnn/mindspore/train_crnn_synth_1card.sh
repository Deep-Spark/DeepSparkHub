#!/bin/bash

pip3 install -r requirements.txt

bash scripts/run_standalone_train.sh synth /home/datasets/mnt/ramdisk/max/90kDICT32px GPU
