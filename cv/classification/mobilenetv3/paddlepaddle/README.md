# MobileNetV3

## Model Description

MobileNetV3 is an efficient convolutional neural network optimized for mobile devices, combining hardware-aware neural
architecture search with novel design techniques. It introduces improved nonlinearities and efficient network structures
to reduce computational complexity while maintaining accuracy. MobileNetV3 achieves state-of-the-art performance in
mobile vision tasks, offering variants for different computational budgets. Its design focuses on minimizing latency and
power consumption, making it ideal for real-time applications on resource-constrained devices like smartphones and
embedded systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.3.0     |  22.12  |

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
## Step 1: Installing
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

**Notice**: modify PaddleClas/ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_small_x1_25.yaml file, modify the datasets
path as yours.

```bash
cd PaddleClas
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_small_x1_25.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
