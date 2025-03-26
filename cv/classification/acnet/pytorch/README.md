# ACNet

## Model Description

ACNet (Asymmetric Convolutional Network) is an innovative CNN architecture that enhances model performance through
Asymmetric Convolution Blocks (ACBs). These blocks use 1D asymmetric convolutions to strengthen standard square
convolution kernels, improving robustness to rotational distortions and reinforcing central kernel structures. ACNet can
be seamlessly integrated into existing architectures, boosting accuracy without additional inference costs. After
training, ACNet converts back to the original architecture, maintaining efficiency. It demonstrates consistent
performance improvements across various models on datasets like CIFAR and ImageNet.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

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
git clone https://github.com/DingXiaoH/ACNet.git
cd ACNet/
git checkout 748fb0c734b41c48eacaacf7fc5e851e33a63ce8
```

## Model Training

```bash
ln -s /path/to/imagenet imagenet_data
rm -rf acnet/acb.py
rm -rf utils/misc.py
mv ../acb.py acnet/
mv ../misc.py utils/

# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' acnet/do_acnet.py
export PYTHONPATH=$PYTHONPATH:.

# One single GPU
export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b acb

# 8 GPUs on one machine
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 acnet/do_acnet.py -a sres18 -b acb
```

## Model Results

| Model | GPU        | ACC                         | FPS      |
|-------|------------|-----------------------------|----------|
| ACNet | BI-V100 ×8 | top1=71.27000,top5=90.00800 | 5.78it/s |

## References

- [ACNet](https://github.com/DingXiaoH/ACNet/tree/748fb0c734b41c48eacaacf7fc5e851e33a63ce8)
