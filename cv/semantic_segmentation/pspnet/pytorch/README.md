# PSPNet

## Model Description

An effective pyramid scene parsing network for complex scene understanding. The global pyramid pooling feature provides additional contextual information.
PSPNet provides a superior framework for pixellevel prediction. The proposed approach achieves state-ofthe-art performance on various datasets. It came first in ImageNet scene parsing challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields the new record of mIoU accuracy 85.4% on PASCAL VOC 2012 and accuracy 80.2% on Cityscapes.

## Model Preparation

### Install Dependencies

```shell
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

### Prepare datasets
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

## Step 2: Training
### Training on single card
```shell
python3 tools/train.py configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py
```

### Training on mutil-cards
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py 8
```

## References
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
