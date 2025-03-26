# Stable Diffusion XL

## Model Description

Stable Diffusion XL is a powerful text-to-image generation model that represents a significant leap in AI-driven visual
creativity. This advanced version builds upon its predecessors with enhanced capabilities for generating
high-resolution, detailed images from textual descriptions. With its larger architecture and improved training
techniques, Stable Diffusion XL excels at producing photorealistic and artistic visuals with greater accuracy and
diversity. The model's ability to interpret complex prompts and generate corresponding images makes it a valuable tool
for creative professionals, designers, and AI enthusiasts.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources


You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

#### Weights

Download the stabilityai/stable-diffusion-xl-base-1.0 from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

Download the madebyollin/sdxl-vae-fp16-fix from [huggingface
page](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

#### Datasets

dataset: download the lambdalabs/pokemon-blip-captions  from [huggingface
page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

### Install Dependencies

```sh
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.27.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install huggingface-hub==0.25.2
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```

## Model Training

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

## References

- [diffusers](https://github.com/huggingface/diffusers)
