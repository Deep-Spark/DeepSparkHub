# DCNv2

## Model description

The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of RCNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.

## Step 1: Installation

DCNv2 model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

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

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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

## Step 3: Training

```bash
# Make soft link to dataset
cd mmdetection/
mkdir -p data/
ln -s /path/to/coco2017 data/coco

# On single GPU
python3 tools/train.py configs/dcnv2/faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/dcnv2/faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py 8
```

## Results

|    GPUs    | FP32     |
| ---------- | -------- |
| BI-V100 x8 | MAP=41.2 |

## Reference

- [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)
