# DeepLab

## Model description

DeepLabv3 is a semantic segmentation architecture that improves upon DeepLabv2 with several modifications. 
To handle the problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. 

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_deeplabv3_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: [torchvision](../../torchvision/pytorch/README.md)
