# BiSeNet

## Model description

A novel Bilateral Segmentation Network (BiSeNet).
First design a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features.
Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain sufficient receptive field.
On top of the two paths, we introduce a new Feature Fusion Module to combine features efficiently. 

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
bash train_bisenet_r18_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
