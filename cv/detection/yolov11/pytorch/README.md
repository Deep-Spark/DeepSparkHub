# YOLOv11

## Model Description

Ultralytics YOLO11 is not just another object detection model; it's a versatile framework designed to cover the entire lifecycle of machine learning models—from data ingestion and model training to validation, deployment, and real-world tracking. Each mode serves a specific purpose and is engineered to offer you the flexibility and efficiency required for different tasks and use-cases.

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

```bash
# Install ultralytics
pip3 install ultralytics
```

## Model Training

```bash
python3 train.py
```

## Model Results

| Model  | GPU        | FP32     |
|--------|------------|----------|
| YOLOv11 | BI-V150 x8 | MAP=39.5 |

## References

- [ultralytics](https://github.com/ultralytics/ultralytics)
