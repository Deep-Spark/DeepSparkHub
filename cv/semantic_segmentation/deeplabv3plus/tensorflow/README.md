# DeepLabV3+

## Model Description

DeepLabV3+ is a state-of-the-art semantic segmentation network. It combines the strengths of DeepLabV3 and a powerful
encoder-decoder architecture. The network employs atrous convolution to capture multi-scale contextual information
effectively. It introduces a novel feature called the "ASPP" module, which utilizes parallel atrous convolutions to
capture fine-grained details and global context simultaneously.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Sign up and login in [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to
download the cityscapes dataset. Specify `/path/to/cityscapes` to your Cityscapes path in later training process.

The Cityscapes dataset path structure should look like:

```bash
Cityscapes
├── leftImg8bit
│   ├── train
│   │   └── aachen
│   │       ├── aachen_000000_000019_leftImg8bit.png
│   │       └── ...
│   └── val
│
├── gtFine
│   ├── train
│   │   └── aachen
│   │       ├── aachen_000000_000019_gtFine_labelTrainIds.png
│   │       └── ...
│   └── val
│
├── license.txt
├── README
├── test.txt
├── train.txt
└── val.txt
```

### Install Dependencies

```bash
pip3 install wandb
pip3 install urllib3==1.26.6
```

## Model Training

Open config folder and set `/path/to/cityscapes` in ./config/cityscapes_resnet50.py.
single gpu:

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python3 trainer.py cityscapes_resnet50 1> train_deeplabv3.log 2> train_deeplabv3_error.log & tail -f train_deeplabv3.log
```

## Model Results

| Model      | GPU     | FPS  | ACC    |
|------------|---------|------|--------|
| DeepLabV3+ | BI-V100 | 6.14 | 77.35% |

## References

- [DeepLabV3-Plus](https://github.com/lattice-ai/DeepLabV3-Plus)
