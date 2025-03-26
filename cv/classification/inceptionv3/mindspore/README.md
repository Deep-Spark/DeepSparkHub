# InceptionV3

## Model Description

InceptionV3 is an advanced convolutional neural network architecture that improves upon previous Inception models with
several key innovations. It introduces factorized convolutions, label smoothing, and an auxiliary classifier to enhance
feature extraction and training stability. The network utilizes batch normalization in side branches to improve gradient
flow and convergence. InceptionV3 achieves state-of-the-art performance in image classification tasks while maintaining
computational efficiency, making it suitable for various computer vision applications requiring high accuracy and robust
feature learning.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

### Install Dependencies

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

## Model Training

```shell
ln -sf $(which python3) $(which python)

# On single GPU
## bash scripts/run_standalone_train_gpu.sh DEVICE_ID DATA_DIR CKPT_PATH
bash scripts/run_standalone_train_gpu.sh /path/to/imagenet/train ./ckpt/

# Multiple GPUs on one machine
## bash scripts/run_distribute_train_gpu.sh DATA_DIR CKPT_PATH
bash scripts/run_distribute_train_gpu.sh /path/to/imagenet/train ./ckpt/

# Evaluation
cd scripts/
DEVICE_ID=0
bash run_eval_gpu.sh $DEVICE_ID /path/to/imagenet/val/ /path/to/checkpoint
```

## Model Results

| Model       | GPU       | epoch | Loss   | ACC                                    | FPS   |
|-------------|-----------|-------|--------|----------------------------------------|-------|
| InceptionV3 | BI-V100×4 | 108   | 3.9033 | 'Top1-Acc': 0.4847, 'Top5-Acc': 0.7405 | 447.2 |

## References

- [mindspore/models](https://gitee.com/mindspore/models/tree/master/official/)
