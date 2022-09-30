# GoogLeNet

## Model description
GoogLeNet is a type of convolutional neural network based on the Inception architecture. It utilises Inception modules, which allow the network to choose between multiple convolutional filter sizes in each block. An Inception network stacks these modules on top of each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid.

## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
```

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model googlenet --batch-size 512
```
### 8 GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenet --model googlenet --batch-size 512 --wd 0.000001
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
