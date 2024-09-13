# YOLOv5

YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## Step 1: Installing packages

```bash
## clone yolov5 and install
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov5/pytorch/
bash init.sh
```

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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

## Training the detector

Train the yolov5 model as follows, the train log is saved in ./runs/train/exp

### On single GPU

```bash
python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights ''
```

### On single GPU (AMP)

```bash
python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5s.yaml --weights '' --amp
```

### Multiple GPUs on one machine

```bash
# eight cards
# YOLOv5s
python3 -m torch.distributed.launch --nproc_per_node 8 \
    train.py \
    --data ./data/coco.yaml \
    --batch-size 64 \
    --cfg ./models/yolov5s.yaml --weights '' \
    --device 0,1,2,3,4,5,6,7

# YOLOv5m
bash run.sh
```

### Multiple GPUs on one machine (AMP)

```bash
# eight cards 
python3 -m torch.distributed.launch --nproc_per_node 8 \
    train.py \
    --data ./data/coco.yaml \
    --batch-size 256 \
    --cfg ./models/yolov5s.yaml --weights '' \
    --device 0,1,2,3,4,5,6,7 --amp
```

## Test the detector

Test the yolov5 model as follows, the result is saved in ./runs/detect:

```bash
python3 detect.py --source ./data/images/bus.jpg --weights yolov5s.pt --img 640

python3 detect.py --source ./data/images/zidane.jpg --weights yolov5s.pt --img 640
```

## Results on BI-V100


| GPUs | FP16 | Batch size | FPS | E2E | mAP@.5 |
| ---- | ---- | ---------- | --- | --- | ------ |
| 1x1  | True | 64         | 81  | N/A | N/A    |
| 1x8  | True | 64         | 598 | 24h | 0.632  |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
| -------------------- | ---------------------------------------- | ----------- | -------- | ---------- | ----------- | ----------------------- | --------- |
| mAP:0.5              | SDK V2.2, bs:128, 8x, AMP                | 1228        | 0.56     | 140\*8     | 0.92        | 27.3\*8                 | 1         |

## Reference

- [YOLOv5](https://github.com/ultralytics/yolov5)
