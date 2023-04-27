# VGG16
## Model description
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

## Step 1: Installing

```bash
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

```bash
cd PaddleClas
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Step 3: Training
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

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
