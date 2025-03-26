# YOLOv6

## Model Description

YOLOv6 is an industrial-grade object detection model that pushes the boundaries of the YOLO series. It incorporates
advanced network design, training strategies, and optimization techniques to achieve state-of-the-art performance.
YOLOv6 offers multiple model sizes for various speed-accuracy trade-offs, excelling in both accuracy and inference
speed. It introduces innovative quantization methods for efficient deployment. The model demonstrates superior
performance compared to other YOLO variants, making it suitable for diverse real-world applications requiring fast and
accurate object detection.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

- data: prepare dataset and specify dataset paths in data.yaml ( [COCO](http://cocodataset.org), [YOLO format coco
  labels](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip) )
- make sure your dataset structure as follows:

```bash
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```

### Install Dependencies

```bash
## install libGL
yum install mesa-libGL

## install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/

## clone yolov6
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov6/pytorch/
pip3 install -r requirements.txt
```

## Model Training

> After training, reporting "AttributeError: 'NoneType' object has no attribute 'python_exit_status'" is a [known
> issue](https://github.com/meituan/YOLOv6/issues/506), add "--workers 0" if you want to avoid.

```bash
# Single gpu training
python3 tools/train.py --batch 32 --conf configs/yolov6s.py --data data/coco.yaml --epoch 300 --name yolov6s_coco

# Multiple gpu training
python3 -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s.py --data data/coco.yaml --epoch 300 --name yolov6s_coco --device 0,1,2,3,4,5,6,7
```

## Model Results

| Model    | Size | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 |
|----------|------|--------------------------|---------------------|
| YOLOv6-S | 640  | 44.3                     | 61.3                |

## References

- [YOLOv6](https://github.com/meituan/YOLOv6)
