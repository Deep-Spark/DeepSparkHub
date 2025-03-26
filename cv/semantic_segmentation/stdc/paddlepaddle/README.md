# STDC

## Model Description

STDC (Short-Term Dense Concatenate network) is an efficient deep learning model for semantic segmentation that reduces
structural redundancy while maintaining high performance. It gradually decreases feature map dimensions and aggregates
them for image representation. The model incorporates a Detail Aggregation module in the decoder to enhance spatial
information learning in low-level layers. By fusing both low-level and deep features, STDC achieves accurate
segmentation results with optimized computational efficiency, making it particularly suitable for real-time
applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

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
yum install mesa-libGL -y
```

### Preprocess Data

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8
python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

## Model Training

Change '/path/to/cityscapes' as your local Cityscapes dataset path.

```bash
data_dir=/path/to/cityscapes
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True

# Single GPU
export CUDA_VISIBLE_DEVICES=0
python3 train.py --config configs/stdcseg/stdc1_seg_cityscapes_1024x512_80k.yml --do_eval --use_vdl --save_interval 500 --save_dir output

# Multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train.py \
       --config configs/stdcseg/stdc1_seg_cityscapes_1024x512_80k.yml \
       --do_eval \
       --use_vdl
```

## Model Results

| GPUs       | Crop Size | Lr schd | FPS   | mIoU   |
|------------|-----------|--------:|-------|--------|
| BI-V100 x8 | 512x1024  |   80000 | 14.36 | 0.7466 |

## References

- [cityscapes](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
