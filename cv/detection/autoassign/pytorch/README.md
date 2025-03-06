# AutoAssign

## Model description

Determining positive/negative samples for object detection is known as label assignment. Here we present an anchor-free detector named AutoAssign. It requires little human knowledge and achieves appearance-aware through a fully differentiable weighting mechanism. During training, to both satisfy the prior distribution of data and adapt to category characteristics, we present Center Weighting to adjust the category-specific prior distributions. To adapt to object appearances, Confidence Weighting is proposed to adjust the specific assign strategy of each instance. The two weighting modules are then combined to generate positive and negative weights to adjust each location's confidence. 


## Step 1: Installing packages

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
```

## Step 2: Preparing datasets

```bash
mkdir -p data 
ln -s /path/to/coco/ ./data

# Prepare resnet50_msra-5891d200.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth -O /root/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth
```

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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

## Step 3: Training

### One single GPU

```bash
python3 tools/train.py configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py
```

### Multiple GPUs on one machine
```bash
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py 8
```

## Reference
[mmdetection](https://github.com/open-mmlab/mmdetection)
