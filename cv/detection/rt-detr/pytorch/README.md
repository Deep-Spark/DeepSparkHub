# RT-DETR

## Model description

RT-DETR is specifically optimized for real-time applications, making it suitable for scenarios where low latency is crucial. It achieves this by incorporating design modifications that improve efficiency without sacrificing accuracy.

## Step 1: Installation

```bash
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

```bash
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

Modify config "img_folder" and "ann_file" locaton in the configuration file(./configs/dataset/coco_detection.yml)

```bash
vim ./configs/dataset/coco_detection.yml
```

## Step 3: Training

### Training on a Single GPU

```bash
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### Training on Multiple GPUs

```bash
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

### Evaluation on Multiple GPUs

```bash
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

## Results

## Reference

[RT-DERT](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
