# DenseNet

## Model Description

DenseNet is an innovative convolutional neural network architecture that introduces dense connections between layers. In
each dense block, every layer receives feature maps from all preceding layers and passes its own features to all
subsequent layers. This dense connectivity pattern improves gradient flow, encourages feature reuse, and reduces
vanishing gradient problems. DenseNet achieves state-of-the-art performance with fewer parameters compared to
traditional CNNs, making it efficient for various computer vision tasks like image classification and object detection.

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
pip install torch torchvision
```

## Model Training

```bash
# One single GPU
python3 train.py --data-path /path/to/imagenet --model densenet201 --batch-size 128

# Multiple GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model densenet201 --batch-size 128
```

## References

- [vision](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)
