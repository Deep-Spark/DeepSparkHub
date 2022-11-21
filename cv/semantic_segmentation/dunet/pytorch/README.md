# DUNet

## Model description

Deformable U-Net (DUNet), which exploits the retinal vessels' local features with a U-shape architecture, in an end to end manner for retinal vessel segmentation.
The DUNet, with upsampling operators to increase the output resolution, is designed to extract context information and enable precise localization by combining low-level feature maps with high-level ones. 
Furthermore, DUNet captures the retinal vessels at various shapes and scales by adaptively adjusting the receptive fields according to vessels' scales and shapes.

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
bash train_dunet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
