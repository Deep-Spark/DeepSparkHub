# FPENet

## Model description

A lightweight feature pyramid encoding network (FPENet) to make a good trade-off between accuracy and speed. 
Specifically, use a feature pyramid encoding block to encode multi-scale contextual features with depthwise dilated convolutions in all stages of the encoder.
A mutual embedding upsample module is introduced in the decoder to aggregate the high-level semantic features and low-level spatial details efficiently. 

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_fpenet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
