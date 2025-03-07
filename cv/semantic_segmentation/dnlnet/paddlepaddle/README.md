# DNLNet

## Model Description

DNLNet (Disentangled Non-Local Neural Networks) is an advanced deep learning model for semantic segmentation that
enhances feature representation through disentangled non-local operations. It separates spatial and channel-wise
relationships in feature maps, enabling more effective long-range dependency modeling. This approach improves the
network's ability to capture contextual information while maintaining computational efficiency. DNLNet demonstrates
superior performance in tasks requiring precise spatial understanding, such as urban scene segmentation, by effectively
aggregating both local and global features.

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

Notice: modify configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml file, modify the datasets path as yours.

```bash
cd PaddleSeg
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=4,5,6,7 # setup 4 available cards
python3 -u -m paddle.distributed.launch train.py \
       --config configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml \
       --do_eval \
       --precision fp16 \
       --amp_level O1
```

## References

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- [Paper](https://arxiv.org/abs/2006.06668)
