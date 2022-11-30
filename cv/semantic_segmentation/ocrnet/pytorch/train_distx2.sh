# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
GPU_NUMS=2

python3 -m torch.distributed.launch --nproc_per_node=$GPU_NUMS tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2.yaml

