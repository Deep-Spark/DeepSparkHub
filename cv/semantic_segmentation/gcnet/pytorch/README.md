# GCNet

## Model description

A Global Context Network, or GCNet, utilises global context blocks to model long-range dependencies in images.
It is based on the Non-Local Network, but it modifies the architecture so less computation is required.
Global context blocks are applied to multiple layers in a backbone network to construct the GCNet.

## Step 1: Installing
### Datasets

- download cityscape from file server or official urls
[Cityscapes](https://www.cityscapes-dataset.com/login/)

```bash
mkdir data/
cd data/
```
- convert_datasets
```bash
python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```
- when done data folder looks like
```bash
data/
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│   │   ├── test
│   │   ├── train
│   │   └── val
    ├── test.lst
    ├── trainval.lst
    └── val.lst
```

### Build Extension
```shell
$ cd /path/to/modelzoo/official/cv/semantic_segmentation/GCNet/pytorch#
$ MMCV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
$ pip3 install -r requirements.txt
```
## Step 2: Training
### Training on single card
```shell
$ bash run_train.sh 
```

### Training on mutil-cards
```shell
$ bash dist_train.sh config gcnet_r50-d8_769x769_40k_cityscapes.py 4
```

## Reference

Ref: https://github.com/LikeLy-Journey/SegmenTron 
