# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
GPU_NUMS=4

python3 -m torch.distributed.launch --nproc_per_node=$GPU_NUMS tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

