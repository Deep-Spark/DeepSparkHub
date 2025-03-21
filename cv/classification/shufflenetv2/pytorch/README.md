# ShuffleNetV2

## Model Description

ShuffleNetv2 is an efficient convolutional neural network designed specifically for mobile devices. It introduces
practical guidelines for CNN architecture design, focusing on direct speed optimization rather than indirect metrics
like FLOPs. The model features a channel split operation and optimized channel shuffle mechanism, improving both
accuracy and inference speed. ShuffleNetv2 achieves state-of-the-art performance in mobile image classification tasks
while maintaining low computational complexity, making it ideal for resource-constrained applications.

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
bash train_shufflenet_v2_x2_0_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification#shufflenet-v2)
