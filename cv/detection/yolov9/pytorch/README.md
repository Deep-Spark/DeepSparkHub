# YOLOv9

## Model Description

YOLOv9 is the latest advancement in the YOLO series, offering state-of-the-art object detection capabilities. It builds
upon previous versions with enhanced architecture and training techniques for improved accuracy and speed. YOLOv9
introduces innovative features that optimize performance across various hardware platforms. The model maintains the YOLO
tradition of real-time detection while delivering superior results in complex scenarios. Its efficient design makes it
suitable for applications requiring fast and accurate object recognition in diverse environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.06  |

## Model Preparation

### Prepare Resources

Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip),
[val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and
[labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have
previously used a different version of YOLO, we strongly recommend that you delete train2017.cache and val2017.cache
files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip)

```bash
bash scripts/get_coco.sh

# make soft link to coco dataset
mkdir -p datasets/
ln -s /PATH/TO/COCO ./datasets/coco
```

### Install Dependencies

```bash
yum install -y mesa-libGL

# get yolov9 code
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9/
pip install -r requirements.txt
pip install Pillow==9.5.0
```

## Model Training

```bash
# Training on a Single GPU
python3 train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15

# Multiple GPU training
torchrun --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15
```

## References

- [YOLOv9](https://github.com/WongKinYiu/yolov9)
