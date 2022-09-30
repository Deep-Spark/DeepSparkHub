# CenterNet

## Model description
Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## Step 1: Installing packages

```bash
$ pip3 install -r requirements.txt
$ # Compile deformable convolutional(DCNv2)
$ cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
$ ./make.sh 
$ # Optional, only required if you are using extrementnet or multi-scale testing
$ # Compile NMS if want to use multi-scale testing or test ExtremeNet. 
$ cd $CenterNet_ROOT/src/lib/external
$ make
```

## Step 2: Preparing datasets

```bash
$ cd /path/to/modelzoo/cv/detection/centernet/pytorch/data 
# Download from homepage of coco: https://cocodataset.org/
```

## Step 3: Training

### On single GPU
```bash
$ cd /path/to/modelzoo/cv/detection/centernet/pytorch/src 
$ # single card
$ python3 main.py ctdet --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0
```

### Multiple GPUs on one machine
```bash
$ cd /path/to/modelzoo/cv/detection/centernet/pytorch/src 
$ bash run.sh
```

## Reference
https://github.com/bubbliiiing/centernet-pytorch
