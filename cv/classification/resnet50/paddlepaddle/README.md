# ResNet50
## Model description
Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Step 1: Installing

```bash
git clone --recursive  https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install -r requirements.txt
```

## Step 2: Download data

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

## Step 3: Run ResNet50

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Results on BI-V100

<div align="center">

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 4 cards     | Acc@1=76.27,FPS=80.37,BatchSize=64   |

</div>
