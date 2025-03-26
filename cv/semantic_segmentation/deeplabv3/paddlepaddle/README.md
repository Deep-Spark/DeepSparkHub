# DeepLabV3

## Model Description

DeepLabV3 is a semantic segmentation architecture that improves upon DeepLabV2 with several modifications. To handle the
problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in
parallel to capture multi-scale context by adopting multiple atrous rates.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

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

### Install Dependencies

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
```

### Preprocess Data

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8
python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

## Model Training

```bash
# Change '/path/to/cityscapes' as your local Cityscapes dataset path
data_dir=/path/to/cityscapes
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml

export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
    --config configs/deeplabv3/deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml \
    --do_eval \
    --use_vdl
```
