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
# Install mmcv
pushd ../../../../toolbox/MMDetection/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
popd

# Install mmdetection
git clone -b v2.25.0 https://gitee.com/open-mmlab/mmdetection.git
pushd  mmdetection
sed -i 's/sklearn/scikit-learn/g' requirements/optional.txt
sed -i 's/github.com/gitee.com/g' requirements/tests.txt
pip3 install -r requirements.txt
pip3 install yapf addict opencv-python
yum install mesa-libGL
python3 setup.py develop
popd

# Install mmrotate

git clone -b v0.3.2 https://gitee.com/open-mmlab/mmrotate.git
cd mmrotate/
sed -i 's/sklearn/scikit-learn/g' requirements/optional.txt
sed -i 's/sklearn/scikit-learn/g' requirements/tests.txt
sed -i 's/python /python3 /g' tools/dist_train.sh
mv ../patch/convex_assigner.py mmrotate/core/bbox/assigners/convex_assigner.py
mv ../patch/bbox_nms_rotated.py mmrotate/core/post_processing/bbox_nms_rotated.py
mv ../patch/schedule_1x.py configs/_base_/schedules/schedule_1x.py
pip3 install -r requirements.txt
pip3 install shapely
python3 setup.py develop

pip3 install urllib3==1.26.6
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
sed -i 's#data/split_1024_dota1_0/#data/split_ss_dota/#g' configs/_base_/datasets/dotav1.py
```


## Step 3: Training


```bash
# On single GPU
python3 tools/train.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py 

# Multiple GPUs on one machine
export USE_GLOOGPU=1
export UMD_ENABLEMEMPOOL=0 
bash tools/dist_train.sh configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py 8
```

## Results

|     GPUs     | ACC | 
|----------| ----------- |
| BI-V100 x8 | MAP=0.8265 |

