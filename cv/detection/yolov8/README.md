## yolov8

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

## Environment
```bash
## Install zlib 1.2.9
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install

## install libGL
yum install mesa-libGL
```

```
pip3 install ultralytics
```

### Download coco2017

```
$ mkdir -p <project_path>/datasets/coco
$ cd <project_path>/datasets/

```

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


##  Training

```bash
python3 test.py
```

|       model       |     GPU     | FP32                                 | 
|-------------------| ----------- | ------------------------------------ |
|       yolov8n     | 8 cards     | MAP=36.3                             |
