# ResNeXt50_32x4d

## Model description
A ResNeXt repeats a building block that aggregates a set of transformations with the same topology. Compared to a ResNet, it exposes a new dimension, cardinality (the size of the set of transformations) , as an essential factor in addition to the dimensions of depth and width.

## Step 1: Installation
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

## Step 2:Preparing datasets

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
set `/path/to/checkpoint` to save the model.
single gpu:
```bash
export CUDA_VISIBLE_DEVICES=0
python3 train.py  \
    --run_distribute=0 \
    --device_target="GPU" \
    --data_path=/path/to/imagenet/train \
    --output_path /path/to/checkpoint
```

multi-gpu:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python3 train.py  \
    --run_distribute=1 \
    --device_target="GPU" \
    --data_path=/path/to/imagenet/train \
    --output_path /path/to/checkpoint
```

validate:
the " model_data_dir " in checkpoint_file_path should look like: `2022-02-02_time_02_22_22`, you should fill in 
the value based on your actual situation.
```bash
python3 eval.py \
    --data_path=/path/to/imagenet/val \
    --device_target="GPU" \
    --checkpoint_file_path=/path/to/checkpoint/model_data_dir/ckpt_0/
```

## Results

| GPUs        | FPS       | ACC(TOP1)    | ACC(TOP5)    |
|-------------|-----------|--------------|--------------|
| BI-V100 x 8 | 109.97    | 78.18%       | 94.03%       |

## Reference
https://gitee.com/mindspore/models/tree/master/research/cv/ResNeXt
