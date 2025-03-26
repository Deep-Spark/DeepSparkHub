# YOLOv7

## Model Description

YOLOv7 is a state-of-the-art real-time object detection model that introduces innovative trainable bag-of-freebies
techniques. It achieves superior accuracy and speed compared to previous YOLO versions and other detectors. YOLOv7
optimizes model architecture, training strategies, and inference efficiency without increasing computational costs. The
model supports various scales for different performance needs and demonstrates exceptional results on COCO benchmarks.
Its efficient design makes it suitable for real-world applications requiring fast and accurate object detection.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

```bash
# clone yolov7
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov7/pytorch/
```

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

Modify the configuration file(data/coco.yaml)

```bash
vim data/coco.yaml
# path: the root of coco data
# train: the relative path of train images
# val: the relative path of valid images
```

The train2017.txt and val2017.txt file you can get from:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
```

The datasets format as follows:

```bash
    coco
        |- iamges
            |- train2017
            |- val2017
        |- labels
            |- train2017
            |- val2017
        |- train2017.txt
        |- val2017.txt

```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

Train the yolov7 model as follows, the train log is saved in ./runs/train/exp.

```bash
# Single GPU training
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# Multiple GPU training
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 64 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

Transfer learning.

- [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt)
- [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt)
- [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt)
- [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt)
- [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt)
- [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

```bash
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```

Inference on video:

```bash
python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

Inference on image:

```bash
python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

## Model Results

| Model  | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> |
|:-------|:---------:|:-----------------:|:------------------------------:|
| YOLOv7 |    640    |       49.4%       |             68.6%              |

## References

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
