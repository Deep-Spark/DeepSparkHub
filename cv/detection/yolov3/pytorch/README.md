# YOLOv3

## Model description

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

## Step 1: Installing packages

```shell
bash setup.sh
```

## Step 2: Preparing data

```bash
bash weights/download_weights.sh
```

```bash
./data/get_coco_dataset.sh
```

## Step 3: Training

### On single GPU

```bash
bash run_training.sh
```

### Multiple GPUs on one machine

```bash
bash run_dist_training.sh
```

## Reference
https://github.com/eriklindernoren/PyTorch-YOLOv3



