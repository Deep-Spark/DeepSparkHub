# DANet

## Model description

A novel framework, the dual attention network (DANet), for natural scene image segmentation. 
It adopts a self-attention mechanism instead of simply stacking convolutions to compute the spatial attention map, which enables the network to capture global information directly.
DANet uses in parallel a position attention module and a channel attention module to capture feature dependencies in spatial and channel domains. 

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

### Preparing datasets

Download the [COCO Dataset](https://cocodataset.org/#home)

### Training on COCO dataset

```shell
bash train_danet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference 

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
