# Attention U-Net: Learning Where to Look for the Pancreas

## Model description

We propose a novel attention gate (AG) model for medical imaging that automatically learns to focus on target structures of varying shapes and sizes. Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while highlighting salient features useful for a specific task. This enables us to eliminate the necessity of using explicit external tissue/organ localisation modules of cascaded convolutional neural networks (CNNs). AGs can be easily integrated into standard CNN architectures such as the U-Net model with minimal computational overhead while increasing the model sensitivity and prediction accuracy. 

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
python3 tools/train.py configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py
```

### Training on mutil-cards
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py 8
```

## Results

| GPUs | Crop Size | Lr schd | FPS  | mIoU |
| ------ | --------- | ------: | --------  |--------------:|
|  BI-V100 x8 | 512x1024  |   160000 | 54.5180      | 69.39 |

## Reference
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
