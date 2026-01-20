# YOLOv5

## Model Description

YOLOv5 is a state-of-the-art object detection model that builds upon the YOLO architecture, offering improved speed and
accuracy. It features a streamlined design with enhanced data augmentation and anchor box strategies. YOLOv5 supports
multiple model sizes (n/s/m/l/x) for different performance needs. The model is known for its ease of use, fast training,
and efficient inference, making it popular for real-time detection tasks across various applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

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

### Install Dependencies

```bash
pip3 install seaborn
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov5/pytorch/
mkdir -p weights
wget -O weights/yolov5s.pt http://files.deepspark.org.cn:880/deepspark/data/checkpoints/yolov5s.pt
mkdir -p datasets
cd datasets
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/coco2017labels.zip
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/coco128.tgz
tar xf coco128.tgz
unzip -q -d ./ coco2017labels.zip
ln -s coco2017/train2017 ./coco/images/
ln -s coco2017/val2017 ./coco/images/
```

## Model Training

```bash
python3 train.py --data coco.yaml
or
python3 train.py --img-size 640 --batch-size 8 \
 --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --data ./data/coco.yaml  --amp ${nonstrict_mode_args} "$@"
```

## Model Results


| GPU        | Batch size | IoU=0.50:0.95 | IoU=0.50 | IoU=0.75 |
|------------|------------|---------------|----------|----------|
| BI-V150 x8 | 8          | 0.365         | 0.546    | 0.400    |

## References

- [YOLOv5](https://github.com/ultralytics/yolov5/tree/850970e081687df6427898948a27df37ab4de5d3)
