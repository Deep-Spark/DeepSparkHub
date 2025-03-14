# Xception

## Model Description

Xception is a deep convolutional neural network that extends the Inception architecture by replacing standard
convolutions with depthwise separable convolutions. This modification significantly reduces computational complexity
while maintaining high accuracy. Xception introduces extreme Inception modules that completely separate channel and
spatial correlations. The architecture achieves state-of-the-art performance in image classification tasks, offering an
efficient alternative to traditional CNNs. Its design is particularly suitable for applications requiring both high
accuracy and computational efficiency.

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
pip3 install torch torchvision
```

## Model Training

```bash
# One single GPU
python3 train.py --data-path /path/to/imagenet --model xception

# Multiple GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model xception
```

## References

- [Xception-PyTorch](https://github.com/tstandley/Xception-PyTorch)
