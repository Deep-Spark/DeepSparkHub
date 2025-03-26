# ResNeXt50_32x4d

## Model Description

ResNeXt50 is an enhanced version of ResNet50 that introduces cardinality as a new dimension alongside depth and width.
It uses grouped convolutions to create multiple parallel transformation paths within each block, improving feature
representation. The 32x4d variant has 32 groups with 4-dimensional transformations. This architecture achieves better
accuracy than ResNet50 with similar computational complexity, making it efficient for image classification tasks.
ResNeXt50's design has influenced many subsequent CNN architectures in computer vision.

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
# Multiple GPUs on one machine
bash train_resnext50_32x4d_amp_dist.sh
```

## References

- [imgclsmob](https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/seresnext.py#L200)
