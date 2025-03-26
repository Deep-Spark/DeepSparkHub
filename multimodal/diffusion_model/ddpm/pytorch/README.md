# DDPM

## Model Description

DDPM (Denoising Diffusion Probabilistic Models) are a class of generative models that learn to generate data by
gradually denoising it through a Markov chain. Inspired by non-equilibrium thermodynamics, DDPMs work by progressively
adding Gaussian noise to data during training and then learning to reverse this process. This approach allows the model
to generate high-quality samples by starting from random noise and iteratively refining it. DDPMs have shown impressive
results in image generation, offering stable training and producing diverse, realistic outputs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p stats && cd stats
```

Download precalculated statistic for dataset:

[cifar10.train.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC)

the dataset structure sholud look like:

```sh
stats
└── cifar10.train.npz
```

### Install Dependencies

```sh
pip3 install -U pip setuptools
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3
yum install -y mesa-libGL 
pip3 install urllib3==1.26.6
```

## Model Training

```sh
cd ../

# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 main.py --train --flagfile ./config/CIFAR10.txt --parallel

# 1 GPU
export CUDA_VISIBLE_DEVICES=0

python3 main.py --train --flagfile ./config/CIFAR10.txt
```

## Step 4: Evaluate

```sh
# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 main.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --notrain --eval --parallel

# 1 GPU
export CUDA_VISIBLE_DEVICES=0

python3 main.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --notrain --eval
```

## Model Results

| Model | GPUs       | FPS       |
|-------|------------|-----------|
| DDPM  | BI-V100 x8 | 1.65 it/s |

![image](images/cifar10_samples.png)

## References

- [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm/tree/master)
