# YOLOv8

## Model Description

YOLOv8 is the latest iteration in the YOLO series, offering state-of-the-art performance in object detection and
tracking. It introduces enhanced architecture and training techniques for improved accuracy and speed. YOLOv8 supports
multiple tasks including instance segmentation, pose estimation, and image classification. The model is designed for
efficiency and ease of use, making it suitable for real-time applications. It maintains the YOLO tradition of fast
inference while delivering superior detection performance across various scenarios.

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
# Install zlib 1.2.9
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install

# Install libGL
yum install mesa-libGL

# Install ultralytics
pip3 install ultralytics
```

## Model Training

```bash
python3 test.py
```

## Model Results

| Model  | GPU        | FP32     |
|--------|------------|----------|
| YOLOv8 | BI-V100 x8 | MAP=36.3 |

## References

- [ultralytics](https://github.com/ultralytics/ultralytics)
