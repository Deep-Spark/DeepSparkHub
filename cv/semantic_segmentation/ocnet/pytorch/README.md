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
bash train_ocnet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron
Ref: [torchvision](../../torchvision/pytorch/README.md)
