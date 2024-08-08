# [CenterMask](https://arxiv.org/abs/1911.06667)2

[[`CenterMask(original code)`](https://github.com/youngwanLEE/CenterMask)][[`vovnet-detectron2`](https://github.com/youngwanLEE/vovnet-detectron2)][[`arxiv`](https://arxiv.org/abs/1911.06667)] [[`BibTeX`](#CitingCenterMask)]

**CenterMask2** is an upgraded implementation on top of [detectron2](https://github.com/facebookresearch/detectron2) beyond original [CenterMask](https://github.com/youngwanLEE/CenterMask) based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

> **[CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667) (CVPR 2020)**<br>
> [Youngwan Lee](https://github.com/youngwanLEE) and Jongyoul Park<br>
> Electronics and Telecommunications Research Institute (ETRI)<br>
> pre-print : https://arxiv.org/abs/1911.06667


<div align="center">
  <img src="https://dl.dropbox.com/s/yg9zr1tvljoeuyi/architecture.png" width="850px" />
</div>

  
  
## Installation
All you need to use centermask2 is [detectron2](https://github.com/facebookresearch/detectron2). It's easy!    
you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).   
Prepare for coco dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).
```bash
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
git clone https://github.com/youngwanLEE/centermask2.git
cd centermask2
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



To train a model, run
For example, to launch CenterMask training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:
```bash
cd centermask2
python3 train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 8
```

