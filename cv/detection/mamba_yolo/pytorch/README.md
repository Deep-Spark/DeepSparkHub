# Mamba-YOLO

## Model description

Mamba-YOLO is an innovative object detection model that integrates State Space Models (SSMs) into the YOLO (You Only Look Once) architecture to enhance performance in complex visual tasks. This integration aims to improve the model's ability to capture global dependencies and process long-range information efficiently.

## Step 1: Installation

```sh
pip3 install seaborn thop timm einops

git clone --depth 1 https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd https://gitee.com/deep-spark/deepsparkhub-GPL/tree/master/cv/detection/mamba-yolo/pytorch

cd selective_scan && pip install . && cd ..
pip install -v -e .
```

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```sh
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

Modify the configuration file(data/coco.yaml)

```sh
vim ultralytics/cfg/datasets/coco.yaml
# path: the root of coco data
# train: the relative path of train images
# val: the relative path of valid images
```

## Step 3: Training

```sh
python3 mbyolo_train.py --task train --data ultralytics/cfg/datasets/coco.yaml \
 --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
--amp  --project ./output_dir/mscoco --name mambayolo_n
```

## Reference

- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO/tree/main)
