# MobileNetV3
## Model description
MobileNetV3 is a convolutional neural network that is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm, and then subsequently improved through novel architecture advances. Advances include (1) complementary search techniques, (2) new efficient versions of nonlinearities practical for the mobile setting, (3) new efficient network design.

## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

```bash
cd paddleclas
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
Notice: modify PaddleClas/ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_small_x1_25.yaml file, modify the datasets path as yours.
```
cd PaddleClas
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_small_x1_25.yaml -o Arch.pretrained=False -o Global.device=gpu
```

## Reference
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)