# SegNet

## Model description

SegNet is a semantic segmentation model.
This core trainable segmentation architecture consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer.
The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network. 
The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. 
The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature maps.
Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling.

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_segnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: [torchvision](../../torchvision/pytorch/README.md)
