#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=8 \
                      --master_port 10002 \
 ncf_32.py data/ml-20m \
  -l  "0.0045" \
  -e   5      \
  -b  "1048576" \
  -b1 "0.25" \
  -b2 "0.5" \
  --eps "1e-8" \
  --valid-batch-size "1048576" \
  --loss-scale 8192  \
  --layers  256 256 128 64 -f 64  \
  --seed 1234 \
  --threshold 0.635  2>&1 |tee ncf_fp32.txt