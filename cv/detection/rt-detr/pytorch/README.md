# RT-DETR

## Model Description

RT-DETR is a real-time variant of the DETR (DEtection TRansformer) model, optimized for efficient object detection. It
maintains the transformer-based architecture of DETR while introducing modifications to reduce latency and improve
speed. RT-DETR achieves competitive accuracy with significantly faster inference times, making it suitable for
applications requiring real-time performance. The model preserves the end-to-end detection capabilities of DETR while
addressing its computational challenges, offering a practical solution for time-sensitive detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.06  |

## Model Preparation

### Prepare Resources

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

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
# Training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml

# Train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml

# Evaluation on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

## References

[RT-DERT](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
