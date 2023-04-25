# FasterNet

## Model description

This is the official Pytorch/PytorchLightning implementation of the paper: <br/>
> [**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**](https://arxiv.org/abs/2303.03667)      
> Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan        
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023*
> 

--- 
We propose a simple yet fast and effective partial convolution (**PConv**), as well as a latency-efficient family of architectures called **FasterNet**.

## Step 1: Installing
### 1. Dependency Setup
Clone this repo and install required packages:
```
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    n01440764/
      n01440764_10026.JPEG
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
```

## Step 2: Training
**Remark**: Training will prompt wondb visualization options, you'll need a W&B account to visualize, choose "3" if you don't need to.

FasterNet-T0 training on ImageNet-1K with a 8-GPU node:
```
# You can change the dataset path '--data_dir' according to your own dataset path !!!
python3 train_test.py -g 0,1,2,3,4,5,6,7 --num_nodes 1 -n 4 -b 4096 -e 2000 \
--data_dir /home/datasets/cv/imagenet --pin_memory --wandb_project_name fasternet \
--model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') --cfg cfg/fasternet_t0.yaml
```

FasterNet-T0 training on ImageNet-1K with a 1-GPU node:
```
# You can change the dataset path '--data_dir' according to your own dataset path !!!
python3 train_test.py -g 0 --num_nodes 1 -n 4 -b 512 -e 2000 \
--data_dir /home/datasets/cv/imagenet --pin_memory --wandb_project_name fasternet \
--model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') --cfg cfg/fasternet_t0.yaml
```

To train other FasterNet variants, `--cfg` need to be changed. You may also want to change the training batch size `-b`.       


## Result

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     |  test_acc1 71.832 val_acc1 71.722    |

## Reference
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) , [poolformer](https://github.com/sail-sg/poolformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [mmdetection](https://github.com/open-mmlab/mmdetection) repositories.
