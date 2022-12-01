# APCNet

## Model descripstion

Adaptive Pyramid Context Network (APCNet) for semantic segmentation. 
APCNet adaptively constructs multi-scale contextual representations with multiple well-designed Adaptive Context Modules (ACMs).
Specifically, each ACM leverages a global image representation as a guidance to estimate the local affinity coefficients for each sub-region.
And then calculates a context vector with these affinities.

## Step 1: Installing

### Install packages

```shell
pip3 install -r requirements.txt
```

### Build Extension

```shell
python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

## Step 2: Prepare Datasets

Download cityscapes from file server or official website [Cityscapes](https://www.cityscapes-dataset.com)

```shell
mkdir -p data/
ln -s ${CITYSCAPES_DATASET_PATH} data/cityscapes
```

## Step 3: Training

**The available configs are as follows:**

```shell
# VOC2012
apcnet_r50-d8_512x512_1k_voc.py
apcnet_r50-d8_512x512_20k_voc

# ADE
apcnet_r50-d8_512x512_80k_ade20k
apcnet_r50-d8_512x512_160k_ade20k
apcnet_r101-d8_512x512_80k_ade20k
apcnet_r101-d8_512x512_160k_ade20k

# CityScapes
apcnet_r50-d8_512x1024_1k_cityscapes
apcnet_r50-d8_512x1024_40k_cityscapes
apcnet_r50-d8_512x1024_80k_cityscapes
apcnet_r50-d8_769x769_40k_cityscapes
apcnet_r50-d8_769x769_80k_cityscapes
apcnet_r101-d8_512x1024_40k_cityscapes
apcnet_r101-d8_512x1024_80k_cityscapes
apcnet_r101-d8_769x769_40k_cityscapes
apcnet_r101-d8_769x769_80k_cityscapes
```


### Training on single card
```shell
bash train.sh <config file> [training args]    # config file can be found in the configs directory 
```

### Training on mutil-cards
```shell
bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

### Example

```shell
bash train_dist.sh configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py 8
```

### Training arguments

```python
# the dir to save logs and models
work-dir: str = None

# the checkpoint file to load weights from
load-from: str = None

# the checkpoint file to resume from
resume-from: str = None

# whether not to evaluate the checkpoint during training
no-validate: bool = False

# (Deprecated, please use --gpu-id) number of gpus to 
# use (only applicable to non-distributed training)
gpus: int = None

# (Deprecated, please use --gpu-id) ids of gpus to use 
# (only applicable to non-distributed training)
gpu-ids: int = None

# id of gpu to use (only applicable to non-distributed training)
gpu-id: int = 0

# random seed
seed: int = None

# Whether or not set different seeds for different ranks
diff_seed: bool = False

# whether to set deterministic options for CUDNN backend.
deterministic: bool = False

# --options is deprecated in favor of --cfg_options' and it 
# will not be supported in version v0.22.0. Override some 
# settings in the used config, the key-value pair in xxx=yyy 
# format will be merged into config file. If the value to be 
# overwritten is a list, it should be like key="[a,b]" or key=a,b 
# It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" 
# Note that the quotation marks are necessary and that no white space 
# is allowed.
options: str = None

# override some settings in the used config, the key-value pair 
# in xxx=yyy format will be merged into config file. If the value 
# to be overwritten is a list, it should be like key="[a,b]" or key=a,b 
# It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" 
# Note that the quotation marks are necessary and that no white 
# space is allowed.
cfg-options: str = None

# job launcher
launcher: str = "none"

# local rank
local_rank: int = 0

# distributed backend
dist_backend: str = None

# resume from the latest checkpoint automatically.
auto-resume: bool = False
```

## Results

### Cityscapes

#### Accuracy

| Method | Backbone | Crop Size | Lr schd | Mem (GB)  | mIoU (BI x 4) |
| ------ | -------- | --------- | ------: | --------  |--------------:|
| APCNet | R-50-D8  | 512x1024  |   40000 | 7.7       |         77.53 |


### VOC2012

#### Accuracy

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | mIoU (BI x 4) |
| ------ | -------- |-----------|--------:|----------|--------------:|
| APCNet | R-50-D8  | 512x512   |    1000 | 20.1     |         66.63 |
| APCNet | R-50-D8  | 512x512   |   20000 | 20.1     |         69.68 |

## Reference
-Ref: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes
-Ref: [Author Results](configs/apcnet/README.md)
-Ref: https://github.com/open-mmlab/mmsegmentation
