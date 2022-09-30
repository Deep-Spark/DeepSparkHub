#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node 8 train.py --data ./data/coco.yaml --batch-size 128 --cfg ./models/yolov5s.yaml --device 0,1,2,3,4,5,6,7
