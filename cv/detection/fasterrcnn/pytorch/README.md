# Faster R-CNN

## Model Description

Faster R-CNN is a state-of-the-art object detection model that introduces a Region Proposal Network (RPN) to generate
region proposals efficiently. It shares convolutional features between the RPN and detection network, enabling nearly
cost-free region proposals. This architecture significantly improves detection speed and accuracy compared to its
predecessors. Faster R-CNN achieves excellent performance on benchmarks like PASCAL VOC and COCO, and serves as the
foundation for many winning entries in computer vision competitions.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
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

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

cd <project_path>/start_scripts
bash init_torch.sh
```

## Model Training

```bash
# On single GPU (AMP)
cd <project_path>/start_scripts
bash train_fasterrcnn_resnet50_amp_torch.sh --dataset coco --data-path /path/to/coco2017

## Multiple GPUs on one machine
cd <project_path>/start_scripts
bash train_fasterrcnn_resnet50_amp_dist_torch.sh --dataset coco --data-path /path/to/coco2017
```

## References

- [vision](https://github.com/pytorch/vision)
