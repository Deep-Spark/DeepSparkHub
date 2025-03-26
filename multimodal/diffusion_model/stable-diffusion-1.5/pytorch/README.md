# Stable Diffusion 1.5

## Model Description

Stable Diffusion 1.5 is an advanced text-to-image generative model that creates high-quality, photorealistic images from
textual descriptions. Building upon its predecessor, this version enhances image generation capabilities through
improved latent diffusion techniques. The model operates in a compressed latent space, enabling efficient processing
while maintaining exceptional visual quality. With its ability to interpret diverse text prompts and generate
corresponding images, Stable Diffusion 1.5 has become a powerful tool for creative applications, AI-assisted design, and
visual content generation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

#### Weights

Download the runwayml/stable-diffusion-v1-5 from [huggingface
page](https://huggingface.co/runwayml/stable-diffusion-v1-5).

#### Datasets

dataset: download the lambdalabs/pokemon-blip-captions  from [huggingface
page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

### Install Dependencies

```bash
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.29.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install -r requirements.txt
pip3 install pillow --upgrade
```

## Model Training

If you have downloaded the weights and dataset, please export the environment variables like below.

```bash
export MODEL_PATH=/path/to/sd_weights
export DATASET_PATH=/path/to/data
```

```bash
# Go to diffusers path
cd ${PROJ_ROOT}/toolbox/diffusers

# Single GPU
bash run_sd_1.5_single.sh

# Multi GPUs
bash run_sd_1.5_multi.sh
```

## Model Results

| Model  | GPUs    | ips_per_device | ips_per_gpu |
| ------ | ------- | -------------- | ----------- |
| SD 1.5 | BI-V150 | 6.76           | 13.5        |

## References

- [diffusers](https://github.com/huggingface/diffusers)
