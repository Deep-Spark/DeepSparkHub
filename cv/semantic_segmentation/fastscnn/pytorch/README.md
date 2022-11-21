# FastSCNN

## Model description

The fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024x2048px) suited to efficient computation on embedded devices with low memory.
The 'learning to downsample' module which computes low-level features for multiple resolution branches simultaneously. 
The network combines spatial detail at high resolution with deep features extracted at lower resolution.

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
bash train_fastscnn_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
