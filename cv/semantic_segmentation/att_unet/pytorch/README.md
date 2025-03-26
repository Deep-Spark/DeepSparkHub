# Attention U-Net

## Model Description

Attention U-Net is an advanced deep learning model for medical image segmentation that integrates attention gates into
the U-Net architecture. These gates automatically learn to focus on target structures of varying shapes and sizes,
suppressing irrelevant regions while highlighting salient features. This eliminates the need for explicit localization
modules, improving model sensitivity and accuracy. Attention U-Net efficiently processes medical images with minimal
computational overhead, making it particularly effective for tasks requiring precise segmentation of complex anatomical
structures.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the
Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure
should look like:

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

### Install Dependencies

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

## Model Training

```shell
# Training on single card
python3 tools/train.py configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py

# Training on mutil-cards
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py 8
```

## Model Results

| GPUs       | Crop Size | Lr schd | FPS     | mIoU  |
|------------|-----------|---------|---------|-------|
| BI-V100 x8 | 512x1024  | 160000  | 54.5180 | 69.39 |

## References

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
