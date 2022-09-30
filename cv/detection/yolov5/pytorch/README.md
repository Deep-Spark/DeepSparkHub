# YOLOv5

YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## Step 1: Installing packages

```shell
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

Download the [COCO Dataset](https://cocodataset.org/#home) 

Modify the configuration file(data/coco.yaml)

    $ vim data/coco.yaml
    $ # path: the root of coco data
    $ # train: the relative path of train images
    $ # val: the relative path of valid images

## Training the detector

Train the yolov5 model as follows, the train log is saved in ./runs/train/exp

### On single GPU

    $ cd yolov5 
    $ python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5m.yaml 

### On single GPU (AMP)

    $ python3 train.py --data ./data/coco.yaml --batch-size 32 --cfg ./models/yolov5m.yaml --amp


### Multiple GPUs on one machine

    $ # eight cards 
    $ python3 -m torch.distributed.launch --nproc_per_node 8 train.py --data ./data/coco.yaml --batch-size 256 --cfg ./models/yolov5m.yaml --device 0,1,2,3,4,5,6,7  # bash run_dist_training.sh

### Multiple GPUs on one machine (AMP)

    $ # eight cards 
    $ python3 -m torch.distributed.launch --nproc_per_node 8 train.py --data ./data/coco.yaml --batch-size 256 --cfg ./models/yolov5m.yaml --device 0,1,2,3,4,5,6,7 --amp


## Test the detector

Test the yolov5 model as follows, the result is saved in ./runs/detect:

    $ cd yolov5
    $ python3 detect.py --source ./data/images/bus.jpg --weight yolov5m.pt --img 640
    $ python3 detect.py --source ./data/images/zidane.jpg --weight yolov5m.pt --img 640


## Results on BI-V100

| GPUs | FP16 | Batch size | FPS | E2E | mAP@.5 |
|------|------|------------|-----|-----|--------|
| 1x1  | True  | 64         | 81  | N/A | N/A    |
| 1x8  | True  | 64         | 598 | 24h | 0.632  |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| mAP:0.5              | SDK V2.2,bs:128,8x,AMP                   | 1228        | 0.56     | 140\*8     | 0.92        | 27.3\*8                 | 1         |


## Reference
https://github.com/ultralytics/yolov5
