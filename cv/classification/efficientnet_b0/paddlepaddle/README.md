# EfficientNetB0

## Model description

This model is the B0 version of the EfficientNet series, whitch can be used for image classification tasks, such as cat and dog classification, flower classification, and so on.

## Step 1: Installation

```bash
git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleClas.git

cd PaddleClas
pip3 install -r requirements.txt
pip3 install paddleclas
pip3 install protobuf==3.20.3
yum install mesa-libGL 
pip3 install urllib3==1.26.15

```


## Step 2: Preparing datasets

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. 

The ImageNet dataset path structure should look like:

```bash
PaddleClas/dataset/ILSVRC2012/
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

**Tips**

For `PaddleClas` training, the image path in train_list.txt and val_list.txt must contain `train/` and `val/` directories:
- train_list.txt: train/n01440764/n01440764_10026.JPEG 0
- val_list.txt: val/n01667114/ILSVRC2012_val_00000229.JPEG 35

```bash
# add "train/" and "val/" to head of lines
sed -i 's#^#train/#g' train_list.txt
sed -i 's#^#val/#g' val_list.txt
```

## Step 3: Training

```bash
# Link your dataset to default location
cd PaddleClas/
ln -s /path/to/imagenet ./dataset/ILSVRC2012

# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch tools/train.py -c ppcls/configs/ImageNet/EfficientNet/EfficientNetB0.yaml

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c ppcls/configs/ImageNet/EfficientNet/EfficientNetB0.yaml
```

## Results

| GPUs| ips | Top1 | Top5 |
| ------ | ----------  |--------------|--------------|
|  BI-V100 x8 |   1065.28      | 0.7683 | 0.9316 |

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5)