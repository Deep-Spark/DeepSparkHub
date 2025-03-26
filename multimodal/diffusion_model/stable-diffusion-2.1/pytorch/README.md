# Stable Diffusion 2.1

## Model Description

Stable Diffusion 2.1 is a state-of-the-art text-to-image generation model that builds upon its predecessors with
enhanced capabilities. This latent diffusion model excels at creating high-resolution, photorealistic images from
textual descriptions. With improved architecture and training techniques, it offers better image quality and prompt
understanding compared to earlier versions. The model operates efficiently in a compressed latent space, making it
accessible for various applications while maintaining exceptional visual fidelity. Stable Diffusion 2.1 has become a
powerful tool for creative professionals and AI enthusiasts alike.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

#### Weights

Download the stabilityai/stable-diffusion-2-1-base from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

#### Datasets

Download the lambdalabs/pokemon-blip-captions  from [huggingface
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
bash run_sd_2.1_single.sh

# Multi GPUs
bash run_sd_2.1_multi.sh
```

## References

- [diffusers](https://github.com/huggingface/diffusers)
