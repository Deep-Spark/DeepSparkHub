# Wave-MLP

## Model Description

Wave-MLP is an innovative vision architecture that represents image tokens as wave functions with amplitude and phase
components. It dynamically modulates token relationships through phase terms, adapting to varying semantic information
in different images. This approach enhances feature aggregation in pure MLP architectures, outperforming traditional
CNNs and transformers in tasks like image classification and object detection. Wave-MLP offers efficient computation
while maintaining high accuracy, making it suitable for various computer vision applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

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
pip install thop timm==0.4.5 torchprofile
git clone https://github.com/huawei-noah/Efficient-AI-Backbones.git
cd Efficient-AI-Backbones/wavemlp_pytorch/
git checkout 25531f7fdcf61e300b47c52ba80973d0af8bb011
```

## Model Training

### WaveMLP_T*

### Multiple GPUs on one machine

```bash
# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' train.py
# change dataset
sed -i "s@from timm.data import Dataset@from timm.data import ImageDataset@" train.py
sed -i "s@dataset_train = Dataset(train_dir)@dataset_train = ImageDataset(train_dir)@" train.py
sed -i "s@dataset_eval = Dataset(eval_dir)@dataset_eval = ImageDataset(eval_dir)@" train.py
sed -i 's/args.max_history/100/g' train.py

python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 train.py /your_path_to/imagenet/ --output /your_path_to/output/  --model WaveMLP_T_dw --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 5 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path 0.1 -b 128
```

## Model Results on BI-V100

### FP16

| Model    | GPU       | precision | batchsize | opt-level | fps     |
|----------|-----------|-----------|-----------|-----------|---------|
| Wave-MLP | BI-V100x8 | FP16      | 128       | O1        | 884.27  |
| Wave-MLP | BI-V100x1 | FP16      | 128       | O1        | 114.76  |
| Wave-MLP | BI-V100x8 | FP32      | 128       | O1        | 1068.15 |
| Wave-MLP | BI-V100x1 | FP32      | 128       | O1        | 140.48  |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 80.1                 | SDK V2.2,bs:256,8x,fp32                  | 1026        | 83.1     | 198\*8     | 0.98        | 29.4\*8                 | 1         |

## References

- [Efficient-AI-Backbones](https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/wavemlp_pytorch/)
