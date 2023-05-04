# CBAM

## Model description
Official PyTorch code for "[CBAM: Convolutional Block Attention Module (ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)"


## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
```

## Step 2: Training

ResNet50 based examples are included. Example scripts are included under ```./scripts/``` directory.
ImageNet data should be included under ```./data/ImageNet/``` with foler named ```train``` and ```val```.

```
# To train with CBAM (ResNet50 backbone)
# For 8 GPUs
python3 train_imagenet.py --ngpu 8 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM ./data/ImageNet
# For 1 GPUs
python3 train_imagenet.py --ngpu 1 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 64 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM ./data/ImageNet
```

## Result

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     |  Prec@1 76.216   fps:83.11           |
| 1 cards     |                  fps:2634.37         |

## Reference

- [MXNet implementation of CBAM with several modifications](https://github.com/bruinxiong/Modified-CBAMnet.mxnet) by [bruinxiong](https://github.com/bruinxiong)
