# UNet++

## Model Description

UNet++ is an advanced deep learning architecture for medical image segmentation, featuring a deeply supervised
encoder-decoder network with nested, dense skip pathways. These redesigned connections reduce the semantic gap between
encoder and decoder feature maps, making the optimization task easier. By enhancing feature map similarity across
network levels, UNet++ improves segmentation accuracy, particularly in complex medical imaging tasks. Its architecture
effectively handles the challenges of precise boundary detection and small object segmentation in medical images.

## Model Preparation

### Prepare Resources

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
python3 tools/train.py configs/unet/unet-s5-d16_pspnet_4xb4-40k_drive-64x64.py

# Training on mutil-cards
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/unet/unet-s5-d16_pspnet_4xb4-40k_drive-64x64.py 8
```

## Model Results

| GPU        | Crop Size | Lr schd | FPS   | mDice |
|------------|-----------|---------|-------|-------|
| BI-V100 x8 | 64x64     | 40000   | 238.9 | 87.52 |

## References

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
