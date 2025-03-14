# DPN92

## Model Description

DPN92 is a dual-path network that combines the strengths of ResNet and DenseNet architectures. It features two parallel
paths: one for feature reuse (like ResNet) and another for feature exploration (like DenseNet). This dual-path approach
enables efficient learning of both shared and new features. DPN92 achieves state-of-the-art performance in image
classification tasks while maintaining computational efficiency. Its unique architecture makes it particularly effective
for tasks requiring both feature preservation and discovery.

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
bash train_dpn92_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
