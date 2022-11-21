# RefineNet

## Model description

RefineNet, a generic multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. 
In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions. 
The individual components of RefineNet employ residual connections following the identity mapping mindset, which allows for effective end-to-end training. 
Further, we introduce chained residual pooling, which captures rich background context in an efficient manner. 

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
bash train_refinenet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
