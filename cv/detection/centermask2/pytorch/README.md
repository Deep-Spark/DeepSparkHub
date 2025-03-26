# CenterMask2

## Model Description

CenterMask2 is an advanced instance segmentation model built on Detectron2, extending the original CenterMask
architecture. It improves mask prediction accuracy by incorporating spatial attention mechanisms and VoVNet backbones.
CenterMask2 enhances object localization and segmentation through its dual-branch design, combining mask and box
predictions effectively. The model achieves state-of-the-art performance on COCO dataset benchmarks, offering efficient
training and inference capabilities. It's particularly effective for complex scenes with overlapping objects and varying
scales.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
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

```bash
mkdir -p <project_path>/datasets/
ln -s /path/to/coco2017 <project_path>/datasets/
```

### Install Dependencies

All you need to use centermask2 is [detectron2](https://github.com/facebookresearch/detectron2). It's easy!

you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

```bash
# Install detectron2
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
```

## Model Training

For example, to launch CenterMask training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:

```bash
git clone https://github.com/youngwanLEE/centermask2.git
cd centermask2
python3 train_net.py --config-file configs/centermask/centermask_R_50_FPN_ms_3x.yaml --num-gpus 8
```

## References

- [CenterMask2](https://github.com/youngwanLEE/centermask2)