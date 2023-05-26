# YOLOv3

## Model description

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

## Step 1: Installing

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip3 install -r requirements.txt
```

## Step 2: Download data

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Run YOLOv3

```bash
# Make sure your dataset path is the same as above
cd PaddleDetection
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o TrainReader.batch_size=16 LearningRate.base_lr=0.002 worker_num=4 --use_vdl=true --eval
```