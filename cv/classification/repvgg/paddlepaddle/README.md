# RepVGG
## Model description
 A simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. 

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
ls -al /home/datasets/imagenet_jpeg/
total 52688
drwxr-xr-x 1002 root root    24576 Mar  1 15:33 train
-rw-r--r--    1 root root 43829433 May 16 07:55 train_list.txt
drwxr-xr-x 1002 root root    24576 Mar  1 15:41 val
-rw-r--r--    1 root root  2144499 May 16 07:56 val_list.txt
-----------------------
# train_list.txt has the following format
train/n01440764/n01440764_10026.JPEG 0
...

# val_list.txt has the following format
val/ILSVRC2012_val_00000001.JPEG 65
-----------------------
```

## Step 3: Run RepVGG

```bash
# Make sure your dataset path is the same as above
#OR
# Modify the image_root of Train mode and Eval mode in the file: PaddleClas/ppcls/configs/ImageNet/RepVGG/RepVGG_A0.yaml

DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/train_list.txt
...
...
  Eval:
    dataset: 
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/val_list.txt

cd PaddleClas
# Link your dataset to default location
ln -s /home/datasets/imagenet_jpeg/ ./dataset/ILSVRC2012
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py -c ppcls/configs/ImageNet/RepVGG/RepVGG_A0.yaml -o Arch.pretrained=False -o Global.device=gpu
```

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     | Acc@1=0.7131                         |
