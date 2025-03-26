# CBAM

## Model Description

CBAM (Convolutional Block Attention Module) is an attention mechanism that enhances CNN feature representations. It
sequentially applies channel and spatial attention to refine feature maps, improving model performance without
significant computational overhead. CBAM helps networks focus on important features while suppressing irrelevant ones,
leading to better object recognition and localization. The module is lightweight and can be easily integrated into
existing CNN architectures, making it a versatile tool for improving various computer vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

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
pip3 install torch
pip3 install torchvision
```

## Model Training

ImageNet data should be included under ```./data/ImageNet/``` with foler named ```train``` and ```val```.

ResNet50 based examples are included. Example scripts are included under ```./scripts/``` directory.

```bash
# To train with CBAM (ResNet50 backbone)
## For 8 GPUs
python3 train_imagenet.py --ngpu 8 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM ./data/ImageNet

## For 1 GPUs
python3 train_imagenet.py --ngpu 1 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 64 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM ./data/ImageNet
```

## Model Results

| Model | GPU        | FP32                      |
|-------|------------|---------------------------|
| CBAM  | BI-V100 x8 | Prec@1 76.216   fps:83.11 |
| CBAM  | BI-V100 x1 | fps:2634.37               |

## References

- [Modified-CBAMnet.mxnet](https://github.com/bruinxiong/Modified-CBAMnet.mxnet) by
