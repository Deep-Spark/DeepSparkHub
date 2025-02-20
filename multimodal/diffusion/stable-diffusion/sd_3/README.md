# Stable Diffusion 3

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text
input.

## Step 1: Preparation

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

### Weights

Download the stabilityai/stable-diffusion-3-medium-diffusers from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).

### Datasets

dataset: download the diffusers/dog-example from [huggingface
page](https://huggingface.co/datasets/diffusers/dog-example).

## Step 2: Installation

```bash
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.29.0-py3-none-any.whl
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/transformers-4.38.1-py3-none-any.whl
pip3 install -r ../diffusers/requirements.txt
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
cd ${PROJ_ROOT}/multimodal/diffusion/stable-diffusion/diffusers

# train
bash run_sd3_dreambooth.sh
```

## Results

| Model | GPUs    | ips_per_device | ips_per_gpu |
| ----- | ------- | -------------- | ----------- |
| SD3   | BI-V150 | 4.34           | 8.68        |

## Reference

- [diffusers](https://github.com/huggingface/diffusers)
