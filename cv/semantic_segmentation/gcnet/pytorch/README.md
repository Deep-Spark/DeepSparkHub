# GCNet

## Model Description

A Global Context Network, or GCNet, utilises global context blocks to model long-range dependencies in images.
It is based on the Non-Local Network, but it modifies the architecture so less computation is required.
Global context blocks are applied to multiple layers in a backbone network to construct the GCNet.

## Step 1: Installing
### Install packages

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# install mmsegmentation
git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git --depth=1
cd mmsegmentation/
pip install -v -e .

pip install ftfy
```

### Datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
mkdir data/
ln -s /path/to/cityscapes data/cityscapes
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

## Step 2: Training
### Training on single card
```shell
python3 tools/train.py configs/gcnet/gcnet_r50-d8_4xb2-40k_cityscapes-769x769.py
```

### Training on mutil-cards
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/gcnet/gcnet_r50-d8_4xb2-40k_cityscapes-769x769.py 8
```

## References
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)