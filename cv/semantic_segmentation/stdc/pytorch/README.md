# STDC

## Model description

We propose a novel and efficient structure named Short-Term Dense Concatenate network (STDC network) by removing structure redundancy. Specifically, we gradually reduce the dimension
of feature maps and use the aggregation of them for image representation, which forms the basic module of STDC
network. In the decoder, we propose a Detail Aggregation module by integrating the learning of spatial information into low-level layers in single-stream manner. Finally,
the low-level features and deep features are fused to predict the final segmentation results. 

## Step 1: Installation

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

## Step 2: Preparing datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure should look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
mkdir -p data/
ln -s /path/to/cityscapes data/
```

## Step 3: Training

### Training on single card
```shell
python3 tools/train.py configs/stdc/stdc1_4xb12-80k_cityscapes-512x1024.py
```

### Training on mutil-cards
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/stdc/stdc1_4xb12-80k_cityscapes-512x1024.py 8
```

## Results

| GPUs | Crop Size | Lr schd | FPS  | mIoU |
| ------ | --------- | ------: | --------  |--------------:|
|  BI-V100 x8 | 512x1024  |   20000 | 39.38     | 70.74 |

## Reference
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
