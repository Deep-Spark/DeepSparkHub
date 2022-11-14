# GoogLeNet

## Model description
GoogLeNet is a type of convolutional neural network based on the Inception architecture. It utilises Inception modules, which allow the network to choose between multiple convolutional filter sizes in each block. An Inception network stacks these modules on top of each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid.

## Step 1: Installing

```bash
git clone --recursive  https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
pip3 install -r requirements.txt
```

## Step 2: Download data

Download the [ImageNet Dataset](https://www.image-net.org/download.php) 

```bash
# IMAGENET PATH as follow:
#ls -al /home/datasets/imagenet_jpeg/
#total 52688
# drwxr-xr-x 1002 root root    24576 Mar  1 15:33 train
# -rw-r--r--    1 root root 43829433 May 16 07:55 train_list.txt
# drwxr-xr-x 1002 root root    24576 Mar  1 15:41 val
# -rw-r--r--    1 root root  2144499 May 16 07:56 val_list.txt
# -----------------------
# train_list.txt has the following format
# n01440764/n01440764_10026.JPEG 0
# ...

# val_list.txt has the following format
# ILSVRC2012_val_00000001.JPEG 65
# -----------------------
```

## Step 3: Run GoogLeNet AMP

```bash
# Modify the file:/paddleclas/ppcls/configs/ImageNet/Inception/GoogLeNet.yaml to add the option of AMP

AMP:
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  # O1: mixed fp16
  level: O1
```

```bash
# Make sure your dataset path is the same as above
cd PaddleClas
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/Inception/GoogLeNet.yaml
```