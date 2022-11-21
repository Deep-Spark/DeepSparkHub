# ENet

## Model description

ENet is a semantic segmentation architecture which utilises a compact encoder-decoder architecture.

Some design choices include:
1. Using the SegNet approach to downsampling y saving indices of elements chosen in max pooling layers, and using them to produce sparse upsampled maps in the decoder.
2. Early downsampling to optimize the early stages of the network and reduce the cost of processing large input frames. The first two blocks of ENet heavily reduce the input size, and use only a small set of feature maps.
3. Using PReLUs as an activation function.
4. Using dilated convolutions.
5. Using Spatial Dropout.

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
bash train_enet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
