# YOLOv3

## Model Description

YOLOv3 is a real-time object detection model that builds upon its predecessors with improved accuracy while maintaining
speed. It uses a deeper backbone network and multi-scale predictions to detect objects of various sizes. YOLOv3 achieves
competitive performance with faster inference times compared to other detectors. It processes images in a single forward
pass, making it efficient for real-time applications. The model balances speed and accuracy, making it popular for
practical detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download VOC PASCAL trainval and test data.

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bash
VOC           # path:  /home/yang/dataset/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```

Download checkpoint.

Exporting loaded COCO weights as TF checkpoint(yolov3_coco.ckpt)[BaiduCloud](https://pan.baidu.com/s/11mwiUy8KotjUVQXqkGGPFQ&shfl=sharepset#list/path=%2F) and save as checkpoint/yolov3_coco_demo.ckpt

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

bash init_tf.sh
```

## Model Training

```bash
bash ./run_training.sh
```

## Model Results

| Model  | GPU     | mAP    | fps      |
|--------|---------|--------|----------|
| YOLOv3 | BI-V100 | 33.67% | 4.34it/s |
