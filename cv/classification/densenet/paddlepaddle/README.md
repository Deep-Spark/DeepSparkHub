# DenseNet
## Model description
A DenseNet is a type of convolutional neural network that utilises dense connections between layers, through Dense Blocks, where we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.

## Step 1: Installation
```bash
git clone --recursive  https://github.com/PaddlePaddle/PaddleClas.git

cd PaddleClas

yum install mesa-libGL -y

pip3 install -r requirements.txt
pip3 install protobuf==3.20.3
pip3 install urllib3==1.26.13

python3 setup.py install
```

## Step 2: Preparing Datasets
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

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012

export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Results

| GPUs        | Top1        | Top5           |ips             |
|-------------|-------------|----------------|----------------|
| BI-V100 x 4 | 0.757       | 0.925          |   171          |

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
