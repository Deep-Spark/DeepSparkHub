# GoogLeNet

## Model description
GoogLeNet is a type of convolutional neural network based on the Inception architecture. It utilises Inception modules, which allow the network to choose between multiple convolutional filter sizes in each block. An Inception network stacks these modules on top of each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid.

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
### One single GPU
```bash
python3 train.py --data-path /path/to/imagenet --model googlenet --batch-size 512
```
### 8 GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model googlenet --batch-size 512 --wd 0.000001
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
