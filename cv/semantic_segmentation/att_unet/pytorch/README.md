# Attention U-Net: Learning Where to Look for the Pancreas

## Model descripstion

We propose a novel attention gate (AG) model for medical imaging that automatically learns to focus on target structures of varying shapes and sizes. Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while highlighting salient features useful for a specific task. This enables us to eliminate the necessity of using explicit external tissue/organ localisation modules of cascaded convolutional neural networks (CNNs). AGs can be easily integrated into standard CNN architectures such as the U-Net model with minimal computational overhead while increasing the model sensitivity and prediction accuracy. 

## Step 1: Installing

### Install packages

```shell
yum install mesa-libGL
pip3 install -r requirements.txt
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
```

### Build Extension

```shell
python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

## Step 2: Prepare Datasets

Download cityscapes from file server or official website [Cityscapes](https://www.cityscapes-dataset.com)

```shell
mkdir -p data/
ln -s ${CITYSCAPES_DATASET_PATH} data/
```

## Step 3: Training

**The available configs are as follows:**

```shell

# CityScapes
attunet_res34_512x1024_160k_cityscapes


### Training on mutil-cards
```shell
bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

### Example

```shell
bash train_dist.sh configs/attunet/attunet_res34_512x1024_160k_cityscapes.py 8
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

| Method | Crop Size | Lr schd | FPS (BI x 8)  | mIoU (BI x 8) |
| ------ | --------- | ------: | --------  |--------------:|
|  ATTUNet  | 512x1024  |   160000 | 54.5180      | 69.39 |

## Reference
-Ref: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes
-Ref: https://github.com/open-mmlab/mmsegmentation
