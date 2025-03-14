# SEResNeXt

## Model Description

SEResNeXt is an advanced convolutional neural network that combines ResNeXt's grouped convolution with
Squeeze-and-Excitation (SE) blocks. It introduces channel attention mechanisms to adaptively recalibrate feature
responses, improving feature representation. The architecture leverages multiple parallel transformation paths within
each block while maintaining computational efficiency. SEResNeXt achieves state-of-the-art performance in image
classification tasks by effectively combining multi-branch transformations with channel-wise attention, making it
particularly suitable for complex visual recognition problems.

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
bash train_seresnext101_32x4d_amp_dist.sh
```

## References

- [imgclsmob](https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/seresnext.py#L214)
