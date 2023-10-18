# Xception41

## Model description

Xception is a convolutional neural network architecture that relies solely on depthwise separable convolution layers.

## Step 1: Installing

```bash
git clone --recursive  https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install scikit-learn easydict visualdl==2.2.0 urllib3==1.26.6
yum install mesa-libGL
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

## Step 3: Run Xception41

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
# Modify the "image_root" in configuration file "./ppcls/configs/ImageNet/Xception/Xception41.yaml" to "./Dataset/ILSVRC2012/train/" and "./Dataset/ILSVRC2012/val/" for training and validation.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py -c ./ppcls/configs/ImageNet/Xception/Xception41.yaml
```

## Results on BI-V100

| GPUs        | TOP1        | TOP5        | ips         |
|:-----------:|:-----------:|:-----------:|:-----------:|
| BI-V100 x 8 |0.783        | 0.941       | 537.04      |
