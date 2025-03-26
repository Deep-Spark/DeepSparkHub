# YOLOv3

## Model Description

YOLOv3 is a real-time object detection model that builds upon its predecessors with improved accuracy while maintaining
speed. It uses a deeper backbone network and multi-scale predictions to detect objects of various sizes. YOLOv3 achieves
competitive performance with faster inference times compared to other detectors. It processes images in a single forward
pass, making it efficient for real-time applications. The model balances speed and accuracy, making it popular for
practical detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```bash
bash weights/download_weights.sh
```

```bash
./data/get_coco_dataset.sh
```

### Install Dependencies

```bash
## clone yolov3 and install
git clone https://gitee.com/deep-spark/deepsparkhub-GPL.git
cd deepsparkhub-GPL/cv/detection/yolov3/pytorch/
bash setup.sh
```

## Model Training

```bash
# On single GPU
bash run_training.sh

# Multiple GPUs on one machine
bash run_dist_training.sh
```

## References

- [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
