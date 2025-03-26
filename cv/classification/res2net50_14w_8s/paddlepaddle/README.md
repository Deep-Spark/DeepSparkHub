# Res2Net50_14w_8s

## Model Description

Res2Net50_14w_8s is a convolutional neural network that enhances ResNet architecture by introducing hierarchical
residual-like connections within individual blocks. It increases the receptive field while reusing feature maps,
improving feature representation. The 14w_8s variant uses 14 width and 8 scales, achieving state-of-the-art performance
in image classification tasks. This architecture effectively balances model complexity and computational efficiency,
making it suitable for various computer vision applications requiring both high accuracy and efficient processing.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

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
cd PaddleClas
# Link your dataset to the default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ./ppcls/configs/ImageNet/Res2Net/Res2Net50_14w_8s.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Model Results

| Model            | GPU        | ACC          | FPS               |
|------------------|------------|--------------|-------------------|
| Res2Net50_14w_8s | BI-V100 x8 | top1: 0.7943 | 338.29 images/sec |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
