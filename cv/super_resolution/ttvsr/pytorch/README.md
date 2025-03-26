# TTVSR

## Model Description

TTVSR (Transformer-based Temporal Video Super-Resolution) is an advanced deep learning model for video enhancement that
leverages transformer architectures to capture long-range frame dependencies. It formulates video frames into
pre-aligned trajectories of visual tokens, enabling effective attention calculation along temporal sequences. This
approach improves video super-resolution by better utilizing temporal information across frames, resulting in higher
quality upscaled videos. TTVSR demonstrates superior performance in handling complex video sequences while maintaining
efficient processing capabilities.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html)

```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
```

### Install Dependencies

```shell
pip3 install -r requirements.txt
```

## Model Training

```shell
# One single GPU
python3 train.py <config file> [training args]   # config file can be found in the configs directory

# Mutiple GPUs on one machine
## bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory
bash dist_train.sh configs/TTVSR_reds4.py 8
```

## Model Results

| GPUs       | FP16  | FPS  | PSNR  |
|------------|-------|------|-------|
| BI-V100 x8 | False | 93.9 | 32.12 |

## References

- [TTVSR](https://github.com/researchmm/TTVSR)
