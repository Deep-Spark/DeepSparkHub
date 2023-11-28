# ACNet (ICCV-2019)
## Model description
As designing appropriate Convolutional Neural Network (CNN) architecture in the context of a given application usually involves heavy human works or numerous GPU hours, the research community is soliciting the architecture-neutral CNN structures, which can be easily plugged into multiple mature architectures to improve the performance on our real-world applications. We propose Asymmetric Convolution Block (ACB), an architecture-neutral structure as a CNN building block, which uses 1D asymmetric convolutions to strengthen the square convolution kernels. For an off-the-shelf architecture, we replace the standard square-kernel convolutional layers with ACBs to construct an Asymmetric Convolutional Network (ACNet), which can be trained to reach a higher level of accuracy. After training, we equivalently convert the ACNet into the same original architecture, thus requiring no extra computations anymore. We have observed that ACNet can improve the performance of various models on CIFAR and ImageNet by a clear margin. Through further experiments, we attribute the effectiveness of ACB to its capability of enhancing the model's robustness to rotational distortions and strengthening the central skeleton parts of square convolution kernels.

## Step 1: Installing

```bash
git clone https://github.com/DingXiaoH/ACNet.git
pip3 install urllib3==1.26.6
cd ACNet
```

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

## Step 2: Training

```bash
ln -s /path/to/imagenet imagenet_data
rm -rf acnet/acb.py
rm -rf utils/misc.py
mv ../acb.py acnet/
mv ../misc.py utils/
export PYTHONPATH=$PYTHONPATH:.
```

### One single GPU
```bash
export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b acb
```
### 8 GPUs on one machine
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 acnet/do_acnet.py -a sres18 -b acb
```

## results

| GPUS      |    acc                        | fps     |
| ----------| ------------------------------|---------|
| BI V100×8 | top1=71.27000,top5=90.00800   | 5.78it/s|

## Reference
https://github.com/DingXiaoH/ACNet