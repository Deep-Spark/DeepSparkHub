# Stable Diffusion

## Model description

Stable Diffusion is a latent text-to-image diffusion model.

## Step 1: Installation

- Install
```bash
wget http://files.deepspark.org.cn:880/deepspark/add-ons/deepspeed-0.14.3+corex.20240718020249-cp310-cp310-linux_x86_64.whl
wget http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.22.0-py3-none-any.whl
wget http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
```
```bash
pip3 install deepspeed-0.14.3+corex.20240718020249-cp310-cp310-linux_x86_64.whl
pip3 install diffusers-0.22.0-py3-none-any.whl
pip3 install transformers-4.38.1-py3-none-any.whl
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```


## Step 2: Preparing datasets
You just need to run the script, and it will automatically download the required data and weights. Or you can manually download the weights and data locally.

dataset: download the lambdalabs/pokemon-blip-captions  from [huggingface page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions). 

weights: download the stabilityai/stable-diffusion-2-1-base from [huggingface page](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). 

weights: download the runwayml/stable-diffusion-v1-5 from [huggingface page](https://huggingface.co/runwayml/stable-diffusion-v1-5). 


weights: download the stabilityai/stable-diffusion-xl-base-1.0 from [huggingface page](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

weights: download the madebyollin/sdxl-vae-fp16-fix from [huggingface page](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

## Step 3: Training

If you have downloaded the weights and data, please import the environment variables like below.
```bash
export MODEL_PATH=/path/to/sd_weights
export DATA_PATH=/path/to/data
export VAE_PATH=/path/to/vae_weights   # only for sdxl
```
### sd2.1 or sd1.5
#### One GPU
```bash
bash run_sd_single.sh
```
#### 8 GPUs
```
bash run_sd_multi.sh
```
### sdxl
#### 8 GPUs
```bash
bash run_sd_xl.sh
```

## Results
### sd2.1

GPUs | FPS
---- | ---
BI-V100 x8 |   ips per gpu=16
```
## Reference

- [diffusers](https://github.com/huggingface/diffusers)

