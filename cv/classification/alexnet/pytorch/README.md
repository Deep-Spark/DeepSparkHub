# AlexNet

## Model description
AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense layers as the basic building blocks.
## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
```
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

## Step 2: Training

```bash
cd start_scripts
```

### One single GPU
```bash
bash train_alexnet_torch.sh --data-path /path/to/imagenet
```
### One single GPU (AMP)
```bash
bash train_alexnet_amp_torch.sh --data-path /path/to/imagenet
```
### 8 GPUs on one machine
```bash
bash train_alexnet_dist_torch.sh --data-path /path/to/imagenet
```
### 8 GPUs on one machine (AMP)
```bash
bash train_alexnet_dist_amp_torch.sh --data-path /path/to/imagenet
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision
