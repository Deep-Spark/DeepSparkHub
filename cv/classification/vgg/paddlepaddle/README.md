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

Notice：if use AMP, modify PaddleClas/ppcls/configs/ImageNet/VGG/VGG16.yaml,

```yaml
AMP:
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  # O1: mixed fp16
  level: O1
```

```bash
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/VGG/VGG16.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
