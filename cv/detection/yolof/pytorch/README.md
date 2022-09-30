# You Only Look One-level Feature

## Model description

This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being 2.5x faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with 7x less training epochs. With an image size of 608x608, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is 13% faster than YOLOv4. Code is available at \url{https://github.com/megvii-model/YOLOF}.

## Step 1: Installing packages

```
pip3 install -r requirements.txt
MMCV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```


## Step 2: Preparing datasets

Download coco datasets from [cocodataset](https://cocodataset.org/#download) 

The data folder like this.
````
data/
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   
│   └── train2017
│   │   
    ├── val2017

````


## Step 2: Training

#### Training on a single GPU

```
bash train.sh
```

#### Training on multiple GPUs

```
bash train_dist.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
for example,
```
bash train_dist.sh configs/yolof/yolof_r50_c5_8x8_1x_coco.py 8
```

## Reference
https://github.com/open-mmlab/mmdetection