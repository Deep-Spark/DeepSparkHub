# EfficientNetB0

## Model Description

EfficientNetB0 is the baseline model in the EfficientNet series, known for its exceptional balance between accuracy and
efficiency. It uses compound scaling to uniformly scale up network width, depth, and resolution, achieving
state-of-the-art performance with minimal computational resources. The model employs mobile inverted bottleneck
convolution (MBConv) blocks with squeeze-and-excitation optimization. EfficientNetB0 is particularly effective for
mobile and edge devices, offering high accuracy in image classification tasks while maintaining low computational
requirements.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

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
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.6 --depth=1
cd PaddleClas
pip3 install -r requirements.txt
python3 setup.py install

```

## Model Training

```bash
# Link your dataset to default location
cd PaddleClas/
ln -s /path/to/imagenet ./dataset/ILSVRC2012

# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/EfficientNet/EfficientNetB0.yaml

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c ppcls/configs/ImageNet/EfficientNet/EfficientNetB0.yaml
```

## Model Results

| Model          | GPU        | ips     | Top1   | Top5   |
|----------------|------------|---------|--------|--------|
| EfficientNetB0 | BI-V100 x8 | 1065.28 | 0.7683 | 0.9316 |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5)
