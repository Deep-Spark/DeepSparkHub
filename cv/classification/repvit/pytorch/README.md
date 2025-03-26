# RepViT

## Model Description

RepViT is an efficient lightweight vision model that combines the strengths of CNNs and Transformers for mobile devices.
It enhances MobileNetV3 architecture with Transformer-inspired design choices, achieving superior performance and lower
latency than lightweight ViTs. RepViT demonstrates state-of-the-art accuracy on ImageNet while maintaining fast
inference speeds, making it ideal for resource-constrained applications. Its pure CNN architecture ensures
mobile-friendliness, with the largest variant achieving 83.7% accuracy at just 2.3ms latency on an iPhone 12.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in the later training process.

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
git clone https://github.com/THU-MIG/RepViT.git
cd RepViT
git checkout 298f42075eda5d2e6102559fad260c970769d34e
pip3 install -r requirements.txt
```

## Model Training

```bash
# On single GPU
python3 main.py --model repvit_m0_9 --data-path /path/to/imagenet --dist-eval

# Multiple GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py --model repvit_m0_9 --data-path /path/to/imagenet --dist-eval
```

Tips:

- Specify your data path and model name!
- Choose "3" when getting the output log below during training.

```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```

## Model Results

| Model  | GPU        | FPS           | ACC          |
|--------|------------|---------------|--------------|
| RepViT | BI-V100 x8 | 1.5984 s / it | Acc@1 78.53% |

## References

- [RepViT](https://github.com/THU-MIG/RepViT/tree/298f42075eda5d2e6102559fad260c970769d34e)
