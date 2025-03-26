# ResNeSt50

## Model Description

ResNeSt50 is a convolutional neural network that enhances ResNet architecture with Split-Attention blocks. It introduces
channel-wise attention mechanisms to improve feature representation, combining multiple feature-map groups with adaptive
feature aggregation. The architecture achieves state-of-the-art performance in image classification tasks by effectively
balancing computational efficiency and model capacity. ResNeSt50's design is particularly suitable for large-scale
visual recognition tasks, offering improved accuracy over standard ResNet variants while maintaining efficient training
and inference capabilities.

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
pip3 install -r requirements.txt
```

## Model Training

Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
# Multiple GPUs on one machine (AMP)
bash train_resnest50_amp_dist.sh
```

## References

- [ResNeSt](https://github.com/zhanghang1989/ResNeSt)
