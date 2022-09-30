# AlexNet

## Model description
AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense layers as the basic building blocks.
## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
```

## Step 2: Training

```bash
cd start_scripts
```

### One single GPU
```bash
bash train_alexnet_torch.sh --data-path /home/datasets/cv/imagenet
```
### One single GPU (AMP)
```bash
bash train_alexnet_amp_torch.sh --data-path /home/datasets/cv/imagenet
```
### 8 GPUs on one machine
```bash
bash train_alexnet_dist_torch.sh --data-path /home/datasets/cv/imagenet
```
### 8 GPUs on one machine (AMP)
```bash
bash train_alexnet_dist_amp_torch.sh --data-path /home/datasets/cv/imagenet
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision
