# YOLOv5

## Model Description

YOLOv5 is a state-of-the-art object detection model that builds upon the YOLO architecture, offering improved speed and
accuracy. It features a streamlined design with enhanced data augmentation and anchor box strategies. YOLOv5 supports
multiple model sizes (n/s/m/l/x) for different performance needs. The model is known for its ease of use, fast training,
and efficient inference, making it popular for real-time detection tasks across various applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

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
## clone yolov5 and install
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov5/pytorch/
bash init.sh
```

Modify the configuration file(data/coco.yaml)

```bash
vim data/coco.yaml
# path: the root of coco data
# train: the relative path of train images
# val: the relative path of valid images
```

## Model Training

Train the yolov5 model as follows, the train log is saved in ./runs/train/exp

```bash
# On single GPU
python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights ''

# On single GPU (AMP)
python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights '' --amp

# Multiple GPUs on one machine
## YOLOv5s
python3 -m torch.distributed.launch --nproc_per_node 8 \
    train.py \
    --data ./data/coco.yaml \
    --batch-size 64 \
    --cfg ./models/yolov5s.yaml --weights '' \
    --device 0,1,2,3,4,5,6,7

## YOLOv5m
bash run.sh

# Multiple GPUs on one machine (AMP)
## eight cards 
python3 -m torch.distributed.launch --nproc_per_node 8 \
    train.py \
    --data ./data/coco.yaml \
    --batch-size 256 \
    --cfg ./models/yolov5s.yaml --weights '' \
    --device 0,1,2,3,4,5,6,7 --amp
```

Test the YOLOv5 model as follows, the results are saved in ./runs/detect.

```bash
python3 detect.py --source ./data/images/bus.jpg --weights yolov5s.pt --img 640

python3 detect.py --source ./data/images/zidane.jpg --weights yolov5s.pt --img 640
```

## Model Results


| GPU        | FP16 | Batch size | FPS | E2E | mAP@.5 |
|------------|------|------------|-----|-----|--------|
| BI-V100 x8 | True | 64         | 598 | 24h | 0.632  |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
| -------------------- | ---------------------------------------- | ----------- | -------- | ---------- | ----------- | ----------------------- | --------- |
| mAP:0.5              | SDK V2.2, bs:128, 8x, AMP                | 1228        | 0.56     | 140\*8     | 0.92        | 27.3\*8                 | 1         |

## References

- [YOLOv5](https://github.com/ultralytics/yolov5)
