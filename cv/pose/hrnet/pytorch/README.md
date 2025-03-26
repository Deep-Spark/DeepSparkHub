# HRNet

## Model Description

HRNet, or High-Resolution Net, is a general purpose convolutional neural network for tasks like semantic segmentation,
object detection and image classification. It is able to maintain high resolution representations through the whole
process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams
one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several stages and
the nth stage contains n streams corresponding to n resolutions. The authors conduct repeated multi-resolution fusions
by exchanging the information across the parallel streams over and over.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

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

### Install Dependencies

```shell
pip3 install -r requirements.txt
```

## Model Training

```shell
export COCO_DATASET_PATH=/path/to/coco2017

# On single GPU
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2

# On single GPU (AMP)
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp

# Multiple GPUs on one machine

python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --dist

# Multiple GPUs on one machine (AMP)
python3 ./tools/train.py --cfg ./configs/coco/w32_512_adam_lr1e-3.yaml --datadir=${COCO_DATASET_PATH} --max_epochs=2 --amp --dist
```

## References

- [HigherHRNet-Human-Pose-Estimation](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
