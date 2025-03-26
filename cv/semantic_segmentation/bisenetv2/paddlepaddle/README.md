# BiSeNetV2

## Model Description

BiSeNet V2 is a two-pathway architecture for real-time semantic segmentation. One pathway is designed to capture the
spatial details with wide channels and shallow layers, called Detail Branch. In contrast, the other pathway is
introduced to extract the categorical semantics with narrow channels and deep layers, called Semantic Branch. The
Semantic Branch simply requires a large receptive field to capture semantic context, while the detail information can be
supplied by the Detail Branch. Therefore, the Semantic Branch can be made very lightweight with fewer channels and a
fast-downsampling strategy. Both types of feature representation are merged to construct a stronger and more
comprehensive feature representation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

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
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
python3 setup.py install
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

# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml --do_eval --use_vdl --save_interval 500 --save_dir output

# Four GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --do_eval \
       --use_vdl
```

| Model     | GPU        | FP32        |
|-----------|------------|-------------|
| BiSeNetV2 | BI-V100 x8 | mIoU=73.45% |
