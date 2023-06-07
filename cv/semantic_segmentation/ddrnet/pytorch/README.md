# DDRNet

## Model descripstion

we proposed a family of efficient backbones specially designed for real-time semantic segmentation. The proposed deep dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module (DAPPM) to enlarge effective receptive fields and fuse multi-scale context based on low-resolution feature maps. Our method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset. 

## Step 1: Installing

### Install packages

```shell
pip3 install -r requirements.txt
yum install mesa-libGL
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
ddrnet_23_slim_512x1024_160k_cityscapes
```

### Training on mutil-cards
```shell
bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

### Example

```shell
bash train_dist.sh configs/ddrnet/ddrnet_23_slim_512x1024_160k_cityscapes.py 4
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

| Method | Crop Size | Lr schd | FPS (BI x 4)  | mIoU (BI x 8) |
| ------ | --------- | ------: | --------  |--------------:|
| DDRNet | 512x1024  |   16000 | 33.085   | 74.8 |

## Reference
-Ref: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes
-Ref: https://github.com/open-mmlab/mmsegmentation
