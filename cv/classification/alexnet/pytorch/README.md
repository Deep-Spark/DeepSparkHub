# AlexNet

## Model Description

AlexNet is a groundbreaking deep convolutional neural network that revolutionized computer vision. It introduced key
innovations like ReLU activations, dropout regularization, and GPU acceleration. With its 8-layer architecture featuring
5 convolutional and 3 fully-connected layers, AlexNet achieved record-breaking performance on ImageNet in 2012. Its
success popularized deep learning and established CNNs as the dominant approach for image recognition. AlexNet's design
principles continue to influence modern neural network architectures in computer vision applications.AlexNet is a
classic convolutional neural network architecture. It consists of convolutions, max pooling and dense layers as the
basic building blocks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

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
pip3 install torch
pip3 install torchvision
```

## Model Training

```bash
cd start_scripts
```

```bash
# One single GPU
bash train_alexnet_torch.sh --data-path /path/to/imagenet

# One single GPU (AMP)
bash train_alexnet_amp_torch.sh --data-path /path/to/imagenet

# 8 GPUs on one machine
bash train_alexnet_dist_torch.sh --data-path /path/to/imagenet

# 8 GPUs on one machine (AMP)
bash train_alexnet_dist_amp_torch.sh --data-path /path/to/imagenet
```

## References

- [vision](https://github.com/pytorch/vision/blob/main/torchvision)
