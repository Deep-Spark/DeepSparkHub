# Stable Diffusion 3

## Model Description

Stable Diffusion 3 represents the latest evolution in text-to-image generation technology, offering unprecedented
capabilities in creating photorealistic images from textual descriptions. This advanced latent diffusion model
incorporates cutting-edge architectural improvements and training techniques, resulting in enhanced image quality and
prompt comprehension. With its ability to generate highly detailed and contextually accurate visuals, Stable Diffusion 3
pushes the boundaries of AI-assisted creativity. The model maintains efficient processing through its latent space
operations while delivering state-of-the-art results in image synthesis and generation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

#### Weights

Download the stabilityai/stable-diffusion-3-medium-diffusers from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).

#### Datasets

dataset: download the diffusers/dog-example from [huggingface
page](https://huggingface.co/datasets/diffusers/dog-example).

### Install Dependencies

```bash
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.29.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install -r ../diffusers/requirements.txt
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

# train
bash run_sd3_dreambooth.sh
```

## Model Results

| Model | GPUs    | ips_per_device | ips_per_gpu |
| ----- | ------- | -------------- | ----------- |
| SD3   | BI-V150 | 4.34           | 8.68        |

## References

- [diffusers](https://github.com/huggingface/diffusers)
