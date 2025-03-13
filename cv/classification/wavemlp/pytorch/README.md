# Wave-MLP

## Model description

In the field of computer vision, recent works show that a pure MLP architecture mainly stacked by fully-connected layers can achieve competing performance with CNN and transformer. An input image of vision MLP is usually split into multiple tokens (patches), while the existing MLP models directly aggregate them with fixed weights, neglecting the varying semantic information of tokens from different images. To dynamically aggregate tokens, we propose to represent each token as a wave function with two parts, amplitude and phase. Amplitude is the original feature and the phase term is a complex value changing according to the semantic contents of input images. Introducing the phase term can dynamically modulate the relationship between tokens and fixed weights in MLP. Based on the wave-like token representation, we establish a novel Wave-MLP architecture for vision tasks. Extensive experiments demonstrate that the proposed Wave-MLP is superior to the state-of-the-art MLP architectures on various vision tasks such as image classification, object detection and semantic segmentation. The source code is available at https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch and https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp.

## Step 1: Installing
```bash
pip install thop timm==0.4.5 torchprofile
git clone https://github.com/huawei-noah/Efficient-AI-Backbones.git
cd Efficient-AI-Backbones/wavemlp_pytorch/
git checkout 25531f7fdcf61e300b47c52ba80973d0af8bb011
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

### WaveMLP_T*:

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

## Results on BI-V100

### FP16

| card-batchsize-AMP opt-level | 1 card | 8 cards |
| :-----| ----: | :----: |
| BI-bs126-O1 | 114.76 | 884.27 |


### FP32

| batch_size | 1 card | 8 cards |
| :-----| ----: | :----: |
| 128 | 140.48 | 1068.15 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 80.1                 | SDK V2.2,bs:256,8x,fp32                  | 1026        | 83.1     | 198\*8     | 0.98        | 29.4\*8                 | 1         |


## Reference
[wavemlp_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/wavemlp_pytorch/)
