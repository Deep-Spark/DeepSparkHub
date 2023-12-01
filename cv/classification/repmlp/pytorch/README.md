# RepMLP

## Model description
RepMLP, a multi-layer-perceptron-style neural network building block for image recognition, which is composed of a series of fully-connected (FC) layers. Compared to convolutional layers, FC layers are more efficient, better at modeling the long-range dependencies and positional patterns, but worse at capturing the local structures, hence usually less favored for image recognition. Construct convolutional layers inside a RepMLP during training and merge them into the FC for inference.

## Step 1: Installation

```bash
pip3 install timm yacs
```

## Step 2: Preparing datasets

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Step 3: Training

```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repmlp.py --arch RepMLPNet-B256 --batch-size 32 --tag my_experiment --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.001 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 10 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256 --data-path [/path/to/imagenet]
```

## Results

|GPUs|FPS|ACC|
|----|---|---|
|BI-V100 x8|319|epoch 40: 64.866%|

## Reference

- [RepMLP](https://github.com/DingXiaoH/RepMLP/tree/3eff13fa0257af28663880d870f327d665f0a8e2)
