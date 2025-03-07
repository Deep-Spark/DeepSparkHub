# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

## Model description

We present UNet++, a new, more powerful architecture for medical image segmentation. Our architecture is essentially
a deeply supervised encoder-decoder network where the encoder and decoder sub-networks are connected through a series of nested, dense skip
pathways. The re-designed skip pathways aim at reducing the semantic
gap between the feature maps of the encoder and decoder sub-networks.
We argue that the optimizer would deal with an easier learning task when
the feature maps from the decoder and encoder networks are semantically
similar.

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

## Step 2: Prepare datasets

If there is `DRIVE` dataset locally

```bash
mkdir -p data/
ln -s ${DRIVE_DATASET_PATH} data/
```

If there is no `DRIVE` dataset locally, you can download `DRIVE` from a file server or
[DRIVE](https://drive.grand-challenge.org/) official website 

```bash
python3 tools/convert_datasets/drive.py /path/to/training.zip /path/to/test.zip
```

## Step 3: Training
### Training on single card
```shell
python3 tools/train.py configs/unet/unet-s5-d16_pspnet_4xb4-40k_drive-64x64.py
```

### Training on mutil-cards
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/unet/unet-s5-d16_pspnet_4xb4-40k_drive-64x64.py 8
```

## Results

| GPUs| Crop Size | Lr schd | FPS | mDice |
| ------ | --------- | ------: | --------  |--------------:|
|  BI-V100 x8 | 64x64  |   40000 | 238.9      | 87.52 |

## Reference
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
