# DFANet

## Model description

An extremely efficient CNN architecture named DFANet for semantic segmentation under resource constraints.
It starts from a single lightweight backbone and aggregates discriminative features through sub-network and sub-stage cascade respectively.
Based on the multi-scale feature propagation, DFANet substantially reduces the number of parameters.
But it still obtains sufficient receptive field and enhances the model learning ability, which strikes a balance between the speed and segmentation performance.

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_dfanet_xceptiona_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
