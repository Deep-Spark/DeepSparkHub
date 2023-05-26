# FCN

## Model description

Fully Convolutional Networks, or FCNs, are an architecture used mainly for semantic segmentation.
They employ solely locally connected layers, such as convolution, pooling and upsampling. 
Avoiding the use of dense layers means less parameters (making the networks faster to train).
It also means an FCN can work for variable image sizes given all connections are local.
The network consists of a downsampling path, used to extract and interpret the context, and an upsampling path, which allows for localization.
FCNs also employ skip connections to recover the fine-grained spatial information lost in the downsampling path.

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
bash train_fcn_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: [torchvision](../../torchvision/pytorch/README.md)
