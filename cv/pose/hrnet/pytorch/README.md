# HRNet

## Model description

HRNet, or High-Resolution Net, is a general purpose convolutional neural network for tasks like semantic segmentation, object detection and image classification. It is able to maintain high resolution representations through the whole process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several stages and the nth stage contains n streams corresponding to n resolutions. The authors conduct repeated multi-resolution fusions by exchanging the information across the parallel streams over and over.

## Step 1: Preparing datasets

Download and extract the [COCO dataset](https://cocodataset.org/#download)
$ cd coco2017
$ unzip -q annotations_trainval2017.zip
$ unzip -q train2017.zip
$ unzip -q val2017.zip
$ unzip -q val2017_mini.zip


## Step 2: Training

### On single GPU
```
$ python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2
```

### On single GPU (AMP)
```
$ python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp
```

### Multiple GPUs on one machine
```
$ python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --dist
```

### Multiple GPUs on one machine (AMP)
```
$ python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp --dist
```

## Reference
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
