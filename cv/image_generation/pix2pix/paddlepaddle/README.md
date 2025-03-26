# Pix2Pix

## Model Description

Pix2Pix uses paired images for image translation, which has two different styles of the same image as input, can be used
for style transfer. Pix2pix is encouraged by cGAN, cGAN inputs a noisy image and a condition as the supervision
information to the generation network, Pix2pix uses another style of image as the supervision information input into the
generation network, so the fake image is related to another style of image which is input as supervision information,
thus realizing the process of image translation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Datasets used by Pix2Pix can be downloaded from [here](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

```bash
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz --no-check-certificate
```

For example, the path structure of facades is as following:

```bash
facades
    ├── test
    ├── train
    └── val
```

### Install Dependencies

```bash
git clone https://github.com/PaddlePaddle/PaddleGAN.git
```

```bash
cd PaddleGAN
pip3 install -r requirements.txt
pip3 install urllib3==1.26.6
yum install mesa-libGL -y
```

## Model Training

```bash
# move facades dataset to data/ 
tar -xzvf facades.tar.gz
mv facades/ data/
# 1 GPU
python3 -u tools/main.py --config-file configs/pix2pix_facades.yaml
```

```bash
# Evaluation
python3 tools/main.py --config-file configs/pix2pix_facades.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## Model Results

| Model   | GPU     | Metric FID | FPS      |
|---------|---------|------------|----------|
| Pix2Pix | BI-V100 | 120.5818   | 16.12240 |

The generated images at epoch 200 is shown below:

![results](results.png)

## References

- [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)
