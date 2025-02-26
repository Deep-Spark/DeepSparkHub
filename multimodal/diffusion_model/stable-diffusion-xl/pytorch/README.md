# Stable Diffusion XL

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text
input.

## Step 1: Preparation

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

### Weights

Download the stabilityai/stable-diffusion-xl-base-1.0 from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

Download the madebyollin/sdxl-vae-fp16-fix from [huggingface
page](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

### Datasets

dataset: download the lambdalabs/pokemon-blip-captions  from [huggingface
page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

## Step 2: Installation

```sh
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.27.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install huggingface-hub==0.25.2
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```

## Step 3: Training

If you have downloaded the weights and dataset, please export the environment variables like below.

```sh
export MODEL_PATH=/path/to/sd_weights
export DATASET_PATH=/path/to/data
export VAE_PATH=/path/to/vae_weights
```

```sh
# Go to diffusers path
cd ${PROJ_ROOT}/toolbox/diffusers

bash run_sd_xl.sh
```

## Reference

- [diffusers](https://github.com/huggingface/diffusers)
