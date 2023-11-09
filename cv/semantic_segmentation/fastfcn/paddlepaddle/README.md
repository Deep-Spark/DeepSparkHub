# FastFCN

## Model description

FastFCN is a fast, lightweight semantic segmentation model that achieves real-time speeds with competitive accuracy. It uses an efficient encoder-decoder architecture and depthwise separable convolutions to reduce computations. The simplified design allows FastFCN to run much faster than prior FCNs while maintaining good segmentation quality. FastFCN demonstrates real-time segmentation is possible with a carefully designed lightweight CNN architecture.

## Step 1: Installing

```bash
git clone --recursive  https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
yum install mesa-libGL
pip3 install scikit-learn easydict visualdl==2.2.0 urllib3==1.26.6
pip3 install -v -e .
```

## Step 2: Download data

Download the ADEChallengeData2016 from [Scene Parsing Challenge 2016](http://sceneparsing.csail.mit.edu/index_challenge.html). Alternatively, you can skip this step because the PaddleSeg framework will automatically download it for you.

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

## Step 3: Run FastFCN

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
# Link your dataset to default location or skip this step
ln -s /path/to/ADEChallengeData2016 ./data/ADEChallengeData2016
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py --config configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml
```

## Results on BI-V100

| GPUs        | mIoU        | ips         |
|:-----------:|:-----------:|:-----------:|
| BI-V100 x 8 |0.436        | 33.68       |
