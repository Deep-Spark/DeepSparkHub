# ResNeSt50

## Model Description

ResNeSt50 is a convolutional neural network that enhances ResNet architecture with Split-Attention blocks. It introduces
channel-wise attention mechanisms to improve feature representation, combining multiple feature-map groups with adaptive
feature aggregation. The architecture achieves state-of-the-art performance in image classification tasks by effectively
balancing computational efficiency and model capacity. ResNeSt50's design is particularly suitable for large-scale
visual recognition tasks, offering improved accuracy over standard ResNet variants while maintaining efficient training
and inference capabilities.

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
git clone --recursive  https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install -r requirements.txt
```

## Model Training

```bash
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/ResNeSt/ResNeSt50.yaml -o Arch.pretrained=False -o Global.device=gpu
```

| Model     | GPU        | FP32         |
|-----------|------------|--------------|
| ResNeSt50 | BI-V100 x4 | Acc@1=0.7677 |
