# HRNet

## Model description

HRNet, or High-Resolution Net, is a general purpose convolutional neural network for tasks like semantic segmentation, object detection and image classification. It is able to maintain high resolution representations through the whole process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several stages and the nth stage contains n streams corresponding to n resolutions. The authors conduct repeated multi-resolution fusions by exchanging the information across the parallel streams over and over.

## Step 1: Installing packages

```shell
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

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

## Step 3: Training

### On single GPU

```shell
export COCO_DATASET_PATH=/path/to/coco2017
```

```shell
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2
```

### On single GPU (AMP)

```shell
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp
```

### Multiple GPUs on one machine

```shell
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --dist
```

### Multiple GPUs on one machine (AMP)

```shell
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp --dist
```

## Reference
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
