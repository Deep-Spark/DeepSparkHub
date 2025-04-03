
# AlphaPose

## Model Description

AlphaPose is an accurate multi-person pose estimator, which is the first open-source system that achieves 70+ mAP (75
mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset. To match poses that correspond to the same person across
frames, we also provide an efficient online pose tracker called Pose Flow. It is the first open-source online pose
tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
# create soft link to coco
mkdir -p /home/datasets/cv/
ln -s /path/to/coco2017 /home/datasets/cv/coco

coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# install requirements
pip3 install seaborn pandas pycocotools matplotlib easydict tensorboardX opencv-python
```

## Model Training

```bash
bash ./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 1
```

## Model Results

| Model     | GPU        | FPS      | ACC         |
|-----------|------------|----------|-------------|
| AlphaPose | BI-V100 x8 | 1.71s/it | acc: 0.8429 |

## References

- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
