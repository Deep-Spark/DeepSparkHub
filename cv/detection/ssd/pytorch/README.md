# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

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

```bash
mkdir -p /home/data/perf/ssd
cd /home/data/perf/ssd
ln -s /path/to/coco/ /home/data/perf/ssd
```

Download backbone.

```bash
cd /home/data/perf/ssd
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
```

## Model Training

```bash
# Multiple GPUs on one machine
cd {deepsparkhub_root_path}/cv/detection/ssd/pytorch/base
source ../iluvatar/config/environment_variables.sh
python3  prepare.py --name iluvatar --data_dir /home/data/perf/ssd
bash run_training.sh --name iluvatar --config V100x1x8 --data_dir /home/data/perf/ssd --backbone_path /home/data/perf/ssd/resnet34-333f7ec4.pth
```

## Model Results

| Model | GPU        | Batch Size | FPS  | Train Epochs | mAP  |
|-------|------------|------------|------|--------------|------|
| SSD   | BI-V100 x8 | 192        | 2858 | 65           | 0.23 |

## References

- [mlcommons](https://github.com/mlcommons/training_results_v0.7/tree/master/NVIDIA/benchmarks/ssd/implementations/pytorch)
