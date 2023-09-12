# MobileNetV3

## Model description
MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1905.02244) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

## Step 1: Installation

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


## Step 2: Preparing datasets

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

```bash
cd ../scripts

# 1 GPU
bash run_train.sh GPU 1 0 /path/to/imagenet/train/

# 8 GPUs
bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 /path/to/imagenet/train/
```
## Step 4: Inference

```bash
bash run_infer.sh GPU /path/to/imagenet/val/ ../train/checkpointckpt_0/mobilenetv3-300_2135.ckpt
```

## Results
<div align="center">
 
| GPUS       | ACC (ckpt107) |  FPS   |
| ---------- | ----------    | ----   |
| BI-V100 ×8 | 0.55          | 378.43 |

</div>

## Reference
- [mindspore/models](https://gitee.com/mindspore/models)