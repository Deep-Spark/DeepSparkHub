# CWD

## Model Description

CWD (Channel-wise Knowledge Distillation) is a novel knowledge distillation method for dense prediction tasks like
semantic segmentation. Unlike traditional spatial distillation, CWD aligns feature maps channel-wise between teacher and
student networks by transforming each channel's feature map into a probability map and minimizing their KL divergence.
This approach focuses on the most salient regions of channel-wise maps, improving distillation efficiency and accuracy.
CWD outperforms spatial distillation methods while requiring less computational cost during training.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the
Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure
sholud look like:

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

```shell
mkdir -p data/
ln -s /path/to/cityscapes data/cityscapes
```

```bash
# Preprocess Data
cd ../
python3 tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

### Install Dependencies

```bash
# install libGL
yum install mesa-libGL

# install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/

# install requirements
pip3 install cityscapesscripts addict opencv-python

# install mmcv
pushd ../../../../toolbox/MMDetection/patch/mmcv/v2.0.0rc4/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
popd

# install mmrazor
cd ../mmrazor
pip3 install -r requirements.txt
pip3 install mmcls==v1.0.0rc6
pip3 install mmsegmentation==v1.0.0
pip3 install mmengine==0.7.3
python3 setup.py develop 
```

## Model Training

```bash
# On single GPU
python3 tools/train.py configs/distill/mmseg/cwd/cwd_logits_pspnet_r101-d8_pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/distill/mmseg/cwd/cwd_logits_pspnet_r101-d8_pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py 8
```

## Model Results

| Model               | GPU        | FP32         |
|---------------------|------------|--------------|
| pspnet_r18(student) | BI-V100 x8 | Miou=  75.32 |

## References

- [mmrazor](https://github.com/open-mmlab/mmrazor)
