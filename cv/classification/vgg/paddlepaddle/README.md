# VGG16
## Model description
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

```bash
cd PaddleClas
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets
Download [ImageNet](https://www.image-net.org/), the path as /home/datasets/imagenet/, then the imagenet path as follows:
```
# IMAGENET PATH as follow:
# drwxr-xr-x 1002 root root    24576 Mar  1 15:33 train
# -rw-r--r--    1 root root 43829433 May 16 07:55 train_list.txt
# drwxr-xr-x 1002 root root    24576 Mar  1 15:41 val
# -rw-r--r--    1 root root  2144499 May 16 07:56 val_list.txt
```

## Step 3: Training
Notice：if use AMP, modify PaddleClas/ppcls/configs/ImageNet/VGG/VGG16.yaml, 
```
AMP:
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  # O1: mixed fp16
  level: O1
```
Notice: modify PaddleClas/ppcls/configs/ImageNet/VGG/VGG16.yaml file, modify the datasets path as yours.
```
cd PaddleClas
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/VGG/VGG16.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)