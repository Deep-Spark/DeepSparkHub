# FasterNet

## Model Description

FasterNet is a high-speed neural network architecture that introduces Partial Convolution (PConv) to optimize
computational efficiency. It achieves superior performance by reducing redundant computations while maintaining feature
learning capabilities. FasterNet is designed for real-time applications, offering an excellent balance between accuracy
and speed. Its innovative architecture makes it particularly effective for mobile and edge devices, where computational
resources are limited. The model demonstrates state-of-the-art results in various computer vision tasks while
maintaining low latency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

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

Clone this repo and install the required packages:

```bash
pip install -r requirements.txt
git clone https://github.com/JierunChen/FasterNet.git
cd FasterNet
git checkout e8fba4465ae912359c9f661a72b14e39347e4954
```

## Model Training

**Remark**: Training will prompt wondb visualization options, you'll need a W&B account to visualize, choose "3" if you
don't need to.

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
                      --data_dir ./imagenet \
                      --pin_memory --wandb_project_name fasternet \
                      --model_ckpt_dir ./model_ckpt/$(date +'%Y%m%d_%H%M%S') \
                      --cfg cfg/fasternet_t0.yaml
```

To train other FasterNet variants, `--cfg` need to be changed. You may also want to change the training batch size `-b`.

## Model Results

| Model     | GPU        | FP32                             |
|-----------|------------|----------------------------------|
| FasterNet | BI-V100 x8 | test_acc1 71.832 val_acc1 71.722 |

## References

- [FasterNet](https://github.com/JierunChen/FasterNet/tree/e8fba4465ae912359c9f661a72b14e39347e4954)
