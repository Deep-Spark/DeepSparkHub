# YOLOv10

## Model Description

YOLOv10 is a cutting-edge object detection model developed by Tsinghua University researchers. It eliminates the need
for non-maximum suppression (NMS) while optimizing model architecture for enhanced efficiency. YOLOv10 achieves
state-of-the-art performance with reduced computational overhead, offering superior accuracy-latency trade-offs across
various model scales. Built on the Ultralytics framework, it addresses limitations of previous YOLO versions, making it
ideal for real-time applications requiring fast and accurate object detection in diverse scenarios.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
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

```bash
# make soft link to coco dataset
mkdir -p datasets/
ln -s /PATH/TO/COCO ./datasets/coco
```

### Install Dependencies

```bash
# CentOS
yum install -y mesa-libGL
# Ubuntu
apt install -y libgl1-mesa-glx

# get yolov10 code
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
sed -i 's/^torch/# torch/g' requirements.txt
pip install -r requirements.txt
```

## Model Training

```bash
# Multiple GPU training
yolo detect train data=coco.yaml model=yolov10n.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

## References

- [YOLOv10](https://github.com/THU-MIG/yolov10)
