# PSPNet

## Model descripstion

An effective pyramid scene parsing network for complex scene understanding. The global pyramid pooling feature provides additional contextual information.
PSPNet provides a superior framework for pixellevel prediction. The proposed approach achieves state-ofthe-art performance on various datasets. It came first in ImageNet scene parsing challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields the new record of mIoU accuracy 85.4% on PASCAL VOC 2012 and accuracy 80.2% on Cityscapes.

## Step 1: Installing

### Install packages

```shell
$ pip3 install -r requirements.txt
```

### Build Extension

```shell
$ python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

### Prepare datasets
**You need to arrange datasets as follow:**
[Datasets](configs/_base_/datasets/datasets.md)

```shell
mkdir -p data
ln -s /path/to/datasets/ data/
```


## Step 2: Training

**There are available configs in ./configs/pspnet/**

### Training on single card
```shell
$ bash train.sh <config file> [training args]    # config file can be found in the configs directory 
```

### Training on mutil-cards
```shell
$ bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

### Example

```shell
bash train_dist.sh pspnet_r50-d8_512x1024_40k_cityscapes 8
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


## Reference
-Ref: https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes
-Ref: [Author Results](configs/pspnet/README.md)
-Ref: https://github.com/open-mmlab/mmsegmentation
