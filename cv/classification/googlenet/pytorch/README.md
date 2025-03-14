# GoogLeNet

## Model Description

GoogLeNet is a pioneering deep convolutional neural network that introduced the Inception architecture. It features
multiple parallel convolutional filters of different sizes within Inception modules, allowing efficient feature
extraction at various scales. The network uses 1x1 convolutions for dimensionality reduction, making it computationally
efficient. GoogLeNet achieved state-of-the-art performance in image classification tasks while maintaining relatively
low computational complexity. Its innovative design has influenced many subsequent CNN architectures in computer vision.

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

## Model Training

```bash
# One single GPU
python3 train.py --data-path /path/to/imagenet --model googlenet --batch-size 512

# 8 GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model googlenet --batch-size 512 --wd 0.000001
```

## References

- [vision](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)
