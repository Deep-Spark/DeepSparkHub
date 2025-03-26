# RepMLP

## Model Description

RepMLP is an innovative neural network architecture that combines the strengths of fully-connected (FC) layers and
convolutional operations. It uses FC layers for efficient long-range dependency modeling while incorporating
convolutional layers during training to capture local structures. Through structural re-parameterization, RepMLP merges
these components into pure FC layers for inference, achieving both high accuracy and computational efficiency. This
architecture is particularly effective for image recognition tasks, offering a novel approach to balance global and
local feature learning.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
imagenet
├── train
│   └── n01440764
│       ├── n01440764_10026.JPEG
│       └── ...
├── train_list.txt
├── val
│   └── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
└── val_list.txt
```

### Install Dependencies

```bash
pip3 install timm yacs
git clone https://github.com/DingXiaoH/RepMLP.git
cd RepMLP
git checkout 3eff13fa0257af28663880d870f327d665f0a8e2
```

## Model Training

```bash
# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main_repmlp.py

# change dataset load
sed -i "s@dataset = torchvision.datasets.ImageNet(root=config.DATA.DATA_PATH, split='train' if is_train else 'val', transform=transform)@dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, prefix), transform=transform)@" data/build.py

python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repmlp.py --arch RepMLPNet-B256 --batch-size 32 --tag my_experiment --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.001 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 10 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256 --data-path [/path/to/imagenet]
```

## Model Results

| Model  | GPU        | FPS | ACC               |
|--------|------------|-----|-------------------|
| RepMLP | BI-V100 x8 | 319 | epoch 40: 64.866% |

## References

- [RepMLP](https://github.com/DingXiaoH/RepMLP/tree/3eff13fa0257af28663880d870f327d665f0a8e2)
