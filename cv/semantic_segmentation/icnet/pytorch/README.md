# ICNet

## Model description

An image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance
ICNet provide in-depth analysis of the framework and introduce the cascade feature fusion unit to quickly achieve high-quality segmentation. 

## Step 1: Installing
### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_icnet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
~                                                           
