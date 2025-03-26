# MobileNetV3_large_x1_0

## Model Description

MobileNetV3_large_x1_0 is an efficient convolutional neural network optimized for mobile devices. It combines
hardware-aware neural architecture search with novel design techniques, including improved nonlinearities and efficient
network structures. This variant offers a balance between accuracy and computational efficiency, achieving 74.9% top-1
accuracy on ImageNet. Its design focuses on reducing latency while maintaining performance, making it suitable for
mobile applications. MobileNetV3_large_x1_0 serves as a general-purpose backbone for various computer vision tasks on
resource-constrained devices.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

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
# Make sure your dataset path is the same as above
cd PaddleClas/
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012

export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Model Results

| Model                  | GPU        | Top1  | Top5  | ips           |
|------------------------|------------|-------|-------|---------------|
| MobileNetV3_large_x1_0 | BI-V100 x4 | 0.749 | 0.922 | 512 samples/s |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
