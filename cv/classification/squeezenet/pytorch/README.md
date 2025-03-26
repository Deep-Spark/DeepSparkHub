# SqueezeNet

## Model Description

SqueezeNet is a lightweight convolutional neural network designed for efficient deployment on resource-constrained
devices. It achieves AlexNet-level accuracy with 50x fewer parameters through innovative "fire modules" that combine 1x1
"squeeze" convolutions with 1x1 and 3x3 "expand" convolutions. The architecture focuses on model compression while
maintaining good classification performance. SqueezeNet is particularly suitable for mobile and embedded applications
where model size and computational efficiency are critical, offering a balance between accuracy and resource
requirements.

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
pip3 install torch torchvision
```

## Model Training

```bash
# One single GPU
python3 train.py --data-path /path/to/imagenet --model squeezenet1_0 --lr 0.001

# Multiple GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenette --model squeezenet1_0 --lr 0.001
```

## References

- [vision](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)
