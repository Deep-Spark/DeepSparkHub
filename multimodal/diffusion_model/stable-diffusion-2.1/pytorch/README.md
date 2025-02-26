# Stable Diffusion 2.1

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text
input.

## Step 1: Preparation

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

### Weights

Download the stabilityai/stable-diffusion-2-1-base from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

### Datasets

Download the lambdalabs/pokemon-blip-captions  from [huggingface
page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

## Step 2: Installation

```bash
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.29.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```

## Step 3: Training

If you have downloaded the weights and dataset, please export the environment variables like below.

```bash
export MODEL_PATH=/path/to/sd_weights
export DATASET_PATH=/path/to/data
```

```bash
# Go to diffusers path
cd ${PROJ_ROOT}/toolbox/diffusers

# Single GPU
bash run_sd_2.1_single.sh

# Multi GPUs
bash run_sd_2.1_multi.sh
```

## Reference

- [diffusers](https://github.com/huggingface/diffusers)
