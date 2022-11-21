# ContextNet

## Model description

ContextNet, a new deep neural network architecture which builds on factorized convolution, network compression and pyramid representation to produce competitive semantic segmentation in real-time with low memory requirement.
ContextNet combines a deep network branch at low resolution that captures global context information efficiently with a shallow branch that focuses on high-resolution segmentation details. 

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
bash train_contextnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
