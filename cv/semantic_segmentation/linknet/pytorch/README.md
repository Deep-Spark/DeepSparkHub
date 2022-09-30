# LinkNet

## Model description

A novel deep neural network architecture which allows it to learn without any significant increase in number of parameters.
The network uses only 11.5 million parameters and 21.2 GFLOPs for processing an image of resolution 3x640x360. 

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_linknet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
