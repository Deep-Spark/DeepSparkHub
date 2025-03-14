# ResNet101

## Model Description

ResNet101 is a deep convolutional neural network with 101 layers, building upon the ResNet architecture's residual
learning framework. It extends ResNet50's capabilities with additional layers for more complex feature extraction. The
model uses skip connections to address vanishing gradient problems, enabling effective training of very deep networks.
ResNet101 achieves state-of-the-art performance in image classification tasks while maintaining computational
efficiency. Its architecture is widely used as a backbone for various computer vision applications, including object
detection and segmentation.

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
bash train_resnet101_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
