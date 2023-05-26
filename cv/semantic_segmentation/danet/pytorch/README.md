# DANet

## Model description

A novel framework, the dual attention network (DANet), for natural scene image segmentation. 
It adopts a self-attention mechanism instead of simply stacking convolutions to compute the spatial attention map, which enables the network to capture global information directly.
DANet uses in parallel a position attention module and a channel attention module to capture feature dependencies in spatial and channel domains. 

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

### Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

### Training on COCO dataset

```shell
bash train_danet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference 

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
