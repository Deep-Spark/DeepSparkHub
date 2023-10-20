# STDC

## Model description

A novel and efficient structure named Short-Term Dense Concatenate network (STDC network) by removing structure redundancy. Specifically, we gradually reduce the dimension
of feature maps and use the aggregation of them for image representation, which forms the basic module of STDC
network. In the decoder, we propose a Detail Aggregation module by integrating the learning of spatial information into low-level layers in single-stream manner. Finally,
the low-level features and deep features are fused to predict the final segmentation results. 

## Step 1: Installing

```bash
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
```

## Step 2: Download data

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
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8

python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

## Step 3: Run STDC

```bash
# Change '/path/to/cityscapes' as your local Cityscapes dataset path
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

## Results

| GPUs | Crop Size | Lr schd | FPS  | mIoU |
| ------ | --------- | ------: | --------  |--------------:|
|  BI-V100 x8 | 512x1024  |   80000 | to be updated     | to be updated |

## Reference
- [cityscapes](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
