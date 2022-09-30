#!/bin/bash

python3 test.py --task val --data data/coco.yaml --weights weights/yolov5s.pt 2>&1 | tee inferencelog.log;
