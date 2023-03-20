# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## Step 1: Installing packages
```
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets
Download the [COCO Dataset](https://cocodataset.org/#home) 

Modify the configuration file(data/coco.yaml)
```
$ vim data/coco.yaml
$ # path: the root of coco data
$ # train: the relative path of train images
$ # val: the relative path of valid images
```
The train2017.txt and val2017.txt file you can get from:
```
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
```
The datasets format as follows:
```
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

## Training

Train the yolov5 model as follows, the train log is saved in ./runs/train/exp

### Single GPU training
```
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```
### Multiple GPU training
```
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

```
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```

## Inference

On video:
```
python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
```
python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```
## Results
| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> |
| :-- | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **49.51%** | **68.84%** |


## Reference
https://github.com/WongKinYiu/yolov7