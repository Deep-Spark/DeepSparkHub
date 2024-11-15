#!/bin/bash


colossalai run --nproc_per_node 16 benchmark.py -c 7b -g  -b 16 --tp 1 --pp 4 --num_steps 50 --model_path /home/model_zoos/nlp/deepseek-moe-16b-base
