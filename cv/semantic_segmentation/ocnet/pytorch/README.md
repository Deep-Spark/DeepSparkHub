# OCNet

## Model description

OCNet: Object Context Network , which focuses on enhancing the role of object information. 
Motivated by the fact that the category of each pixel is inherited from the object it belongs to, we define the object context for each pixel as the set of pixels that belong to the same category as the given pixel in the image. 
We use a binary relation matrix to represent the relationship between all pixels, where the value one indicates the two selected pixels belong to the same category and zero otherwise. 
We propose to use a dense relation matrix to serve as a surrogate for the binary relation matrix.
The dense relation matrix is capable to emphasize the contribution of object information as the relation scores tend to be larger on the object pixels than the other pixels. 
We propose an efficient interlaced sparse self-attention scheme to model the dense relations between any two of all pixels via the combination of two sparse relation matrices. 

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
bash train_ocnet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
