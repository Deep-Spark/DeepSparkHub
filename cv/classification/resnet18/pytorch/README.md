# ResNet18

## Model Description

ResNet18 is a lightweight convolutional neural network with 18 layers, featuring residual connections that enable
efficient training of deep networks. It introduces skip connections that bypass layers, addressing vanishing gradient
problems and allowing for better feature learning. ResNet18 achieves strong performance in image classification tasks
while maintaining computational efficiency. Its compact architecture makes it suitable for applications with limited
resources, serving as a backbone for various computer vision tasks like object detection and segmentation.

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
bash train_resnet18_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
