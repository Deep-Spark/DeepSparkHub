# Wide_ResNet101_2

## Model Description

Wide_ResNet101_2 is an enhanced version of Wide_ResNet101 that further increases network width while maintaining
residual connections. It uses wider residual blocks with more filters per layer, enabling richer feature representation.
This architecture achieves superior performance in image classification tasks by balancing increased capacity with
efficient training. Wide_ResNet101_2 demonstrates improved accuracy over standard ResNet variants while maintaining
computational efficiency, making it suitable for complex visual recognition tasks requiring high performance.

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
bash train_wide_resnet101_2_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
