# Oriented RepPoints

## Model Description

Oriented RepPoints is an innovative object detection model designed for aerial imagery, where objects often appear in
arbitrary orientations. It uses adaptive points representation to capture geometric information of non-axis aligned
instances, offering more precise detection than traditional bounding box approaches. The model incorporates three
oriented conversion functions for accurate classification and localization, along with a quality assessment scheme to
handle cluttered backgrounds. It achieves state-of-the-art performance on aerial datasets like DOTA and HRSC2016.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

#### Get the DOTA dataset

The DOTA dataset can be downloaded from [here](https://captain-whu.github.io/DOTA/dataset.html).
The data structure is as follows:

```bash
mmrotate/data/DOTA/
├── test
│   └── images
├── train
│   ├── images
│   ├── labelTxt
│   │   └── trainset_reclabelTxt
│   └── labelTxt-v1.5
└── val
    ├── images
    ├── labelTxt
    │   └── valset_reclabelTxt
    └── labelTxt-v1.5
```

#### Split the DOTA dataset

Please crop the original images into 1024×1024 patches with an overlap of 200.

```bash
python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```

#### Change root path in base config

Please change `data_root` in `configs/_base_/datasets/dotav1.py` to split DOTA dataset.

```bash
sed -i 's#data/split_ss_dota1_5/#data/split_ss_dota/#g' configs/_base_/datasets/dotav15.py
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# Install mmdetection
pip install mmdet==3.3.0

# Install mmrotate
git clone -b v1.0.0rc1 https://gitee.com/open-mmlab/mmrotate.git --depth=1
cd mmrotate/
pip install -v -e .
sed -i 's/python /python3 /g' tools/dist_train.sh
sed -i 's/3.1.0/3.4.0/g' mmrotate/__init__.py
sed -i 's@points_range\s*=\s*torch\.arange\s*(\s*points\.shape\[0\]\s*)@&.to(points.device)@' mmrotate/models/task_modules/assigners/convex_assigner.py
sed -i 's/from collections import Sequence/from collections.abc import Sequence/g' mmrotate/models/detectors/refine_single_stage.py
```

## Model Training

```bash
# On single GPU
python3 tools/train.py configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py 8
```

## Model Results

| Model              | GPU        | ACC        |
|--------------------|------------|------------|
| Oriented RepPoints | BI-V100 x8 | MAP=0.8265 |

## References

- [mmrotate](https://github.com/open-mmlab/mmrotate)
