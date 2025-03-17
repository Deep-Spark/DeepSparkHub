# PP-LCNet

## Model Description

PP-LCNet is a lightweight CPU-optimized neural network designed for efficient inference on edge devices. It leverages
MKLDNN acceleration strategies to enhance performance while maintaining low latency. The architecture achieves
state-of-the-art accuracy for lightweight models in image classification tasks and performs well in downstream computer
vision applications like object detection and semantic segmentation. PP-LCNet's design focuses on maximizing accuracy
with minimal computational overhead, making it ideal for resource-constrained environments requiring fast and efficient
inference.

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

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location
ln -s /path/to/imagenet ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Model Results

| Model        | GPU        | Crop Size | FPS  | TOP1 Accuracy |
|--------------|------------|-----------|------|---------------|
| PPLCNet_x1_0 | BI-V100 x4 | 224x224   | 2537 | 0.7062        |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
