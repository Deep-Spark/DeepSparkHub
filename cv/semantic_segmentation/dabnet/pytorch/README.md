# DabNet

## Model description

A novel Depthwise Asymmetric Bottleneck (DAB) module, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. 
Based on the DAB module, design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation.
It creates sufficient receptive field and densely utilizes the contextual information. 

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
bash train_dabnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
