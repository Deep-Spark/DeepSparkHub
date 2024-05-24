# YOLOv9

## Model description

YOLOv9 is a state-of-the-art object detection algorithm that belongs to the YOLO (You Only Look Once) family of models. It is an improved version of the original YOLO algorithm with better accuracy and performance.

## Step 1: Installation

```bash
yum install -y mesa-libGL
```

## Step 2: Preparing datasets

```bash
bash scripts/get_coco.sh
```

Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete train2017.cache and val2017.cache files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip)

## Step 3: Training

```bash
# make soft link to coco dataset
mkdir -p datasets/
ln -s /PATH/TO/COCO ./datasets/coco

# get yolov9 code
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9/
pip install -r requirements.txt
pip install Pillow==9.5.0
```

### Training on a Single GPU

```bash
python3 train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15
```

### Multiple GPU training

```bash
torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15
```

## Results

## Reference

[YOLOv9](https://github.com/WongKinYiu/yolov9)
