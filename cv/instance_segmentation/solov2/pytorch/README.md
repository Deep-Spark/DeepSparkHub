# SOLOV2

## Model Description

SOLOv2 is an enhanced instance segmentation model that builds upon SOLO's approach by dynamically learning mask heads
conditioned on object locations. It decouples mask prediction into kernel and feature branches for improved efficiency.
SOLOv2 introduces Matrix NMS, a faster non-maximum suppression technique that processes masks in parallel. This
architecture achieves state-of-the-art performance in both speed and accuracy, with a lightweight version running at
31.3 FPS. It serves as a strong baseline for various instance-level recognition tasks beyond segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
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
apt install -y libgl1-mesa-glx

# install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 --depth=1
cd mmdetection
pip install -v -e .

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# Install others
pip3 install yapf==0.31.0 urllib3==1.26.18
```

## Model Training

```bash
# Make soft link to dataset
cd mmdetection/
mkdir -p data/
ln -s /path/to/coco2017 data/coco

sed -i 's/python /python3 /g' tools/dist_train.sh

# On single GPU
python3 tools/train.py configs/solov2/solov2_r50_fpn_1x_coco.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_coco.py 8
```

## Model Results

| Model  | GPUs       | FPS            |
|--------|------------|----------------|
| SOLOV2 | BI-V100 x8 | 21.26 images/s |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection)
