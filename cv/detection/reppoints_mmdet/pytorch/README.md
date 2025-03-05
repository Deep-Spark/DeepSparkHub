# RepPoints

## Model description

Modern object detectors rely heavily on rectangular bounding boxes, such as anchors, proposals and the final predictions, to represent objects at various recognition stages. The bounding box is convenient to use but provides only a coarse localization of objects and leads to a correspondingly coarse extraction of object features. In this paper, we present RepPoints(representative points), a new finer representation of objects as a set of sample points useful for both localization and recognition. Given ground truth localization and recognition targets for training, RepPoints learn to automatically arrange themselves in a manner that bounds the spatial extent of an object and indicates semantically significant local areas. They furthermore do not require the use of anchors to sample a space of bounding boxes. We show that an anchor-free object detector based on RepPoints can be as effective as the state-of-the-art anchor-based detection methods, with 46.5 AP and 67.4 AP50 on the COCO test-dev detection benchmark, using ResNet-101 model.

## Step 1: Installation
RepPoints model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.
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
python3 tools/train.py configs/reppoints/reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/reppoints/reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py 8
```

## Results

|     GPUs    | FP32     | 
| ----------- | -------- |
| BI-V100 x8  | MAP=43.2 |

## Reference

- [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490)
