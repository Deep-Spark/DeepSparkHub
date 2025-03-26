# OCRNet

## Model Description

OCRNet (Object Contextual Representation Network) is a deep learning model for semantic segmentation that enhances
pixel-level understanding by incorporating object context information. It learns object regions from ground-truth
segmentation and aggregates pixel representations within these regions. By computing relationships between pixels and
object regions, OCRNet augments each pixel's representation with contextual information from relevant objects. This
approach improves segmentation accuracy, particularly in complex scenes, by better capturing object boundaries and
contextual relationships between different image elements.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

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

### Install Dependencies

```bash
git clone -b release/2.8 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3
pip3 install urllib3==1.26.13
python3 setup.py install

yum install -y mesa-libGL
```

### Preprocess Data

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8
python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","

# CityScapes PATH as follow:
tree -L 2 /path/to/cityscapes
/path/to/cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── test.txt
├── train.txt
└── val.txt
```

## Model Training

```bash
# Change '/path/to/cityscapes' as your local Cityscapes dataset path
data_dir=/path/to/cityscapes
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml

# Set up environment variables
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True

# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml --do_eval --use_vdl


# Four GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml  \
       --do_eval \
       --use_vdl
```

## Model Results

| GPU        | IPS              | ACC         |
|------------|------------------|-------------|
| BI-V100 x4 | 2.12 samples/sec | mIoU=0.8120 |

## References

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
