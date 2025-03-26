# ResNeXt50_32x4d

## Model Description

ResNeXt50 is an enhanced version of ResNet50 that introduces cardinality as a new dimension alongside depth and width.
It uses grouped convolutions to create multiple parallel transformation paths within each block, improving feature
representation. The 32x4d variant has 32 groups with 4-dimensional transformations. This architecture achieves better
accuracy than ResNet50 with similar computational complexity, making it efficient for image classification tasks.
ResNeXt50's design has influenced many subsequent CNN architectures in computer vision.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
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

Install OpenMPI and mesa-libGL

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar -xvf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7
./configure --prefix=/usr/local/bin --with-orte
make all
make install
vim ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
source ~/.bashrc
yum install openssh-server, openssh-clients
yum install mesa-libGL
```

## Model Training

set `/path/to/checkpoint` to save the model.

```bash
# Single gpu
export CUDA_VISIBLE_DEVICES=0
python3 train.py  \
    --run_distribute=0 \
    --device_target="GPU" \
    --data_path=/path/to/imagenet/train \
    --output_path /path/to/checkpoint

# Multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python3 train.py  \
    --run_distribute=1 \
    --device_target="GPU" \
    --data_path=/path/to/imagenet/train \
    --output_path /path/to/checkpoint
```

The " model_data_dir " in checkpoint_file_path should look like: `2022-02-02_time_02_22_22`, you should fill in
the value based on your actual situation.

```bash
# Evaluation
export CUDA_VISIBLE_DEVICES=0
python3 eval.py \
    --data_path=/path/to/imagenet/val \
    --device_target="GPU" \
    --checkpoint_file_path=/path/to/checkpoint/model_data_dir/ckpt_0/
```

## Model Results

| Model     | GPU        | FPS    | ACC(TOP1) | ACC(TOP5) |
|-----------|------------|--------|-----------|-----------|
| ResNeXt50 | BI-V100 x8 | 109.97 | 78.18%    | 94.03%    |

## References

- [ResNeXt](https://gitee.com/mindspore/models/tree/master/research/cv/ResNeXt)
