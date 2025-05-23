# ECA MobileNet V2

## Model Description

ECA MobileNet V2 is an efficient convolutional neural network that combines MobileNet V2's lightweight architecture with
an Efficient Channel Attention (ECA) module. The ECA module enhances feature representation by adaptively recalibrating
channel-wise feature responses without dimensionality reduction. This integration improves model performance while
maintaining computational efficiency, making it suitable for mobile and edge devices. ECA MobileNet V2 achieves better
accuracy than standard MobileNet V2 with minimal additional parameters, making it ideal for resource-constrained image
classification tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
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
bash train_eca_mobilenet_v2_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
