# CenterMask2

CenterMask2 is an upgraded implementation on top of detectron2 beyond original CenterMask based on maskrcnn-benchmark.

## Step 1: Installation

All you need to use centermask2 is [detectron2](https://github.com/facebookresearch/detectron2). It's easy!

you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

```bash
# Install detectron2
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
```

## Step 2: Preparing datasets

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

## Step 3: Training

For example, to launch CenterMask training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:

```bash
git clone https://github.com/youngwanLEE/centermask2.git
cd centermask2
python3 train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 8
```

## Reference

- [CenterMask2](https://github.com/youngwanLEE/centermask2)