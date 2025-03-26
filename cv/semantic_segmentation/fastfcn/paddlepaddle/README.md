# FastFCN

## Model Description

FastFCN is a fast, lightweight semantic segmentation model that achieves real-time speeds with competitive accuracy. It
uses an efficient encoder-decoder architecture and depthwise separable convolutions to reduce computations. The
simplified design allows FastFCN to run much faster than prior FCNs while maintaining good segmentation quality. FastFCN
demonstrates real-time segmentation is possible with a carefully designed lightweight CNN architecture.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Download the ADEChallengeData2016 from [Scene Parsing Challenge
2016](http://sceneparsing.csail.mit.edu/index_challenge.html). Alternatively, you can skip this step because the
PaddleSeg framework will automatically download it for you.

The ADEChallengeData2016 dataset path structure should look like:

```bash
ADEChallengeData2016
├── annotations
│   └── training
│       ├── ADE_train_00000001.png
│       └── ...
├── images
│   └── training
│       ├── ADE_train_00000001.jpg
│       └── ...
├── objectInfo150.txt
└── sceneCategories.txt
```

### Install Dependencies

```bash
git clone -b release/2.9 --recursive  https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
yum install -y mesa-libGL
pip3 install scikit-learn easydict visualdl==2.2.0 urllib3==1.26.6
pip3 install -v -e .
```

## Model Training

```bash
# Make sure your dataset path is the same as above
cd PaddleClas/

# Link your dataset to the default location or skip this step
ln -s /path/to/ADEChallengeData2016 ./data/ADEChallengeData2016

# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py --config configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml

# Eval
python3 tools/val.py  --config configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml --model_path output/path/to/model.pdparams
```

## Model Results

| GPUs        | mIoU   | Acc    | Kappa  | Dice  | ips   |
|-------------|--------|--------|--------|-------|-------|
| BI-V100 x 8 | 0.4312 | 0.8083 | 0.7935 | 0.570 | 33.68 |

## References

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
