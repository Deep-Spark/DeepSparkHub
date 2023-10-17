# FasterNet

## Model description

This is the official Pytorch/PytorchLightning implementation of the paper: <br/>
> [**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**](https://arxiv.org/abs/2303.03667)      
> Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan        
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023*
> 

We propose a simple yet fast and effective partial convolution (**PConv**), as well as a latency-efficient family of architectures called **FasterNet**.

## Step 1: Installation
Clone this repo and install the required packages:
```bash
pip install -r requirements.txt
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
**Remark**: Training will prompt wondb visualization options, you'll need a W&B account to visualize, choose "3" if you don't need to.

FasterNet-T0 training on ImageNet with a 8-GPU node:

```bash
# You can change the dataset path '--data_dir' according to your own dataset path !!!
python3 train_test.py -g 0,1,2,3,4,5,6,7 --num_nodes 1 -n 4 -b 4096 -e 2000 \
                      --data_dir /path/to/imagenet \
                      --pin_memory --wandb_project_name fasternet \
                      --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') \
                      --cfg cfg/fasternet_t0.yaml
```

FasterNet-T0 training on ImageNet-1K with a 1-GPU node:

```bash
# You can change the dataset path '--data_dir' according to your own dataset path !!!
python3 train_test.py -g 0 --num_nodes 1 -n 4 -b 512 -e 2000 \
                      --data_dir /path/to/imagenet \
                      --pin_memory --wandb_project_name fasternet \
                      --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') \
                      --cfg cfg/fasternet_t0.yaml
```

To train other FasterNet variants, `--cfg` need to be changed. You may also want to change the training batch size `-b`.       

## Results

| GPUs        | FP32                                |
| ----------- | ------------------------------------ |
| BI-V100 x8  |  test_acc1 71.832 val_acc1 71.722    |

## Reference

- [timm](https://github.com/rwightman/pytorch-image-models)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
