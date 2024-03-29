# Res2Net50_14w_8s

## Model description
Res2Net is modified from the source code of ResNet. The main function of Res2Net is to add hierarchical connections within the block and indirectly increase the receptive field while reusing the feature map.
## Step 1: Installation

```bash
git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 urllib3==1.26.6
yum install -y mesa-libGL
```

## Step 2: Preparing datasets

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
cd PaddleClas
# Link your dataset to the default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ./ppcls/configs/ImageNet/Res2Net/Res2Net50_14w_8s.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Results

| GPUs | ACC | FPS
| ---------- | ------ | ---
| BI-V100 x8 | top1: 0.7943 | 338.29 images/sec

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
