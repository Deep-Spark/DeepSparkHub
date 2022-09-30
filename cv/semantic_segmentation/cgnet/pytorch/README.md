# CGNet

## Model description

A novel Context Guided Network (CGNet), which is a light-weight and efficient network for semantic segmentation. 
Under an equivalent number of parameters, the proposed CGNet significantly outperforms existing segmentation networks. 
Specifically, without any post-processing and multi-scale testing, the proposed CGNet achieves 64.8% mean IoU on Cityscapes with less than 0.5 M parameters.

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_cgnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
