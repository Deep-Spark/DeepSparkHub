# MobileNetV3

## Model Description

MobileNetV3 is an efficient convolutional neural network optimized for mobile devices, combining hardware-aware neural
architecture search with novel design techniques. It introduces improved nonlinearities and efficient network structures
to reduce computational complexity while maintaining accuracy. MobileNetV3 achieves state-of-the-art performance in
mobile vision tasks, offering variants for different computational budgets. Its design focuses on minimizing latency and
power consumption, making it ideal for real-time applications on resource-constrained devices like smartphones and
embedded systems.

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
# Install requirements
pip3 install easydict
yum install mesa-libGL

# Install openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

## Model Training

```bash
cd ../scripts

# 1 GPU
bash run_train.sh GPU 1 0 /path/to/imagenet/train/

# 8 GPUs
bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 /path/to/imagenet/train/

# Inference
bash run_infer.sh GPU /path/to/imagenet/val/ ../train/checkpointckpt_0/mobilenetv3-300_2135.ckpt
```

## Model Results

| Model       | GPU        | ACC (ckpt107) | FPS    |
|-------------|------------|---------------|--------|
| MobileNetV3 | BI-V100 ×8 | 0.55          | 378.43 |
|             |

## References

- [mindspore/models](https://gitee.com/mindspore/models)
