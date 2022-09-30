# EncNet

## Model description

The Context Encoding Module, which captures the semantic context of scenes and selectively highlights class-dependent featuremaps. 
The Context Encoding Module significantly improves semantic segmentation results with only marginal extra computation cost over FCN.

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_encnet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
