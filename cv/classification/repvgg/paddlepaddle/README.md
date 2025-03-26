# RepVGG

## Model Description

RepVGG is a simple yet powerful convolutional neural network architecture that combines training-time multi-branch
topology with inference-time VGG-like simplicity. It uses structural re-parameterization to convert complex training
models into efficient inference models composed solely of 3x3 convolutions and ReLU activations. This approach achieves
state-of-the-art performance in image classification tasks while maintaining high speed and efficiency. RepVGG's design
is particularly suitable for applications requiring both high accuracy and fast inference, making it ideal for
real-world deployment scenarios.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py -c ppcls/configs/ImageNet/RepVGG/RepVGG_A0.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Model Results

| Model  | GPU        | FP32         |
|--------|------------|--------------|
| RepVGG | BI-V100 x8 | Acc@1=0.6990 |
