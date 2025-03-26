# UNet

## Model Description

A network and training strategy that relies on the strong use of data augmentation to use the available annotated
samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding
path that enables precise localization.

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
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
```

### Preprocess Data

```bash
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8
python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

## Model Training

Notice: modify configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml file, modify the datasets path as yours.
The training is use AMP model.

```bash
cd PaddleSeg
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
       --config configs/unet/unet_cityscapes_1024x512_160k.yml \
       --do_eval \
       --use_vdl \
       --save_dir output_unet_amp \
       --precision fp16 \
       --amp_level O1
```

## References

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
