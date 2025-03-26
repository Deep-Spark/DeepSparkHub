# VGG16

## Model Description

VGG is a classic convolutional neural network architecture known for its simplicity and depth. It uses small 3x3
convolutional filters stacked in multiple layers, allowing for effective feature extraction. The architecture typically
includes 16 or 19 weight layers, with VGG16 being the most popular variant. VGG achieved state-of-the-art performance in
image classification tasks and became a benchmark for subsequent CNN architectures. Its uniform structure and deep
design have influenced many modern deep learning models in computer vision.

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

```bash
# Set data path
export DATA_PATH=/path/to/imagenet

# Multiple GPUs on one machine
bash train_vgg16_amp_dist.sh
```

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)
