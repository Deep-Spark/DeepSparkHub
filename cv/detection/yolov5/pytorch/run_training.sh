#!/bin/bash

python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml
