# InceptionV3

## Model description
InceptionV3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifier to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).

## Step 1: Installation

```bash
yum install -y mesa-libGL
pip3 install -r requirements.txt
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/openmpi/bin:$PATH
```

## Step 2: Preparing Datasets
Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
imagenet
├── train
│   └── n01440764
│       ├── n01440764_10026.JPEG
│       └── ...
├── train_list.txt
├── val
│   └── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
└── val_list.txt
```


## Step 3: Training
```shell
ln -sf $(which python3) $(which python)
```

### On single GPU
```shell
bash scripts/run_standalone_train_gpu.sh DEVICE_ID DATA_DIR CKPT_PATH
# example: bash scripts/run_standalone_train_gpu.sh /path/to/imagenet/train ./ckpt/ 
```

### Multiple GPUs on one machine
```shell
bash scripts/run_distribute_train_gpu.sh DATA_DIR CKPT_PATH
# example: bash scripts/run_distribute_train_gpu.sh /path/to/imagenet/train ./ckpt/ 
```

### Use checkpoint to eval
```shell
cd scripts/
DEVICE_ID=0
bash run_eval_gpu.sh $DEVICE_ID /path/to/imagenet/val/ /path/to/checkpoint
```

## Results
| GPUS      |    ACC (epoch 108) | FPS |
| ----------| --------------------------| ----- | 
| BI V100×4 | 'Loss': 3.9033, 'Top1-Acc': 0.4847, 'Top5-Acc': 0.7405 | 447.2 |


## Reference
- [MindSpore Models](https://gitee.com/mindspore/models/tree/master/official/)