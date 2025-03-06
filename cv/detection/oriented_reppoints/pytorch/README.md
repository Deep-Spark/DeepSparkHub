# Oriented RepPoints

## Model description

In contrast to the generic object, aerial targets are often non-axis aligned with arbitrary orientations having
the cluttered surroundings. Unlike the mainstreamed approaches regressing the bounding box orientations, this paper
proposes an effective adaptive points learning approach to aerial object detection by taking advantage of the adaptive
points representation, which is able to capture the geometric information of the arbitrary-oriented instances.
To this end, three oriented conversion functions are presented to facilitate the classification and localization
with accurate orientation. Moreover, we propose an effective quality assessment and sample assignment scheme for
adaptive points learning toward choosing the representative oriented reppoints samples during training, which is
able to capture the non-axis aligned features from adjacent objects or background noises. A spatial constraint is
introduced to penalize the outlier points for roust adaptive learning. Experimental results on four challenging
aerial datasets including DOTA, HRSC2016, UCAS-AOD and DIOR-R, demonstrate the efficacy of our proposed approach.

## Step 1: Installation

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

## Step 2: Preparing datasets

### Get the DOTA dataset

The dota dataset can be downloaded from [here](https://captain-whu.github.io/DOTA/dataset.html).
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

### Split dota dataset

Please crop the original images into 1024×1024 patches with an overlap of 200.

```bash
python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python3 tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```

### Change root path in base config

Please change `data_root` in `configs/_base_/datasets/dotav1.py` to split DOTA dataset.

```bash
sed -i 's#data/split_ss_dota1_5/#data/split_ss_dota/#g' configs/_base_/datasets/dotav15.py
```

## Step 3: Training

```bash
# On single GPU
python3 tools/train.py configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_1x_dota.py 8
```

## Results

|     GPUs     | ACC | 
|----------| ----------- |
| BI-V100 x8 | MAP=0.8265 |

## Reference
[mmrotate](https://github.com/open-mmlab/mmrotate)