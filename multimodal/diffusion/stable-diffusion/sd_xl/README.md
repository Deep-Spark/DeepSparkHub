# Stable Diffusion XL

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Step 1: Preparation

You just need to run the script, and it will automatically download the required data and weights. Or you can manually download the weights and data locally.

### Weights

Download the stabilityai/stable-diffusion-xl-base-1.0 from [huggingface page](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

Download the madebyollin/sdxl-vae-fp16-fix from [huggingface page](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

### Datasets

dataset: download the lambdalabs/pokemon-blip-captions  from [huggingface page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

## Step 2: Installation

```bash
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/deepspeed-0.14.3+corex.20240718020249-cp310-cp310-linux_x86_64.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.22.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```

## Step 3: Training

If you have downloaded the weights and dataset, please export the environment variables like below.

```bash
export MODEL_PATH=/path/to/sd_weights
export DATASET_PATH=/path/to/data
export VAE_PATH=/path/to/vae_weights
```

```bash
# Go to diffusers path
cd ${PROJ_ROOT}/multimodal/diffusion/stable-diffusion/diffusers

bash run_sd_xl.sh
```

## Results

| Model | GPUs    | ips_per_device | ips_per_gpu |
| ----- | ------- | -------------- | ----------- |
| SD XL | BI-V150 |                |             |

## Reference

- [diffusers](https://github.com/huggingface/diffusers)
