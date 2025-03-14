# Xception

## Model Description

Xception is a deep convolutional neural network that extends the Inception architecture by replacing standard
convolutions with depthwise separable convolutions. This modification significantly reduces computational complexity
while maintaining high accuracy. Xception introduces extreme Inception modules that completely separate channel and
spatial correlations. The architecture achieves state-of-the-art performance in image classification tasks, offering an
efficient alternative to traditional CNNs. Its design is particularly suitable for applications requiring both high
accuracy and computational efficiency.

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `./PaddleClas/dataset/` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
ILSVRC2012
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

Tips: for `PaddleClas` training, the image path in train_list.txt and val_list.txt must contain `train/` and `val/`
directories:

- train_list.txt: train/n01440764/n01440764_10026.JPEG 0
- val_list.txt: val/n01667114/ILSVRC2012_val_00000229.JPEG 35

```bash
# add "train/" and "val/" to head of lines
sed -i 's#^#train/#g' train_list.txt
sed -i 's#^#val/#g' val_list.txt
```

### Install Dependencies

```bash
git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install scikit-learn easydict visualdl==2.2.0 urllib3==1.26.6
yum install -y mesa-libGL
```

## Model Training

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py -c ./ppcls/configs/ImageNet/Xception/Xception41.yaml
```

## Model Results

| Model    | GPU        | TOP1  | TOP5  | ips    |
|----------|------------|-------|-------|--------|
| Xception | BI-V100 x8 | 0.783 | 0.941 | 537.04 |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
