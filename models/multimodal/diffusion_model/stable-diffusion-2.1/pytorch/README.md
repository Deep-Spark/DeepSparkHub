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
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

You just need to run the script, and it will automatically download the required data and weights. Or you can manually
download the weights and data locally.

#### Weights

Download the stabilityai/stable-diffusion-2-1-base from [huggingface
page](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

```bash
mkdir -p data/model_zoo/stabilityai/stable-diffusion-2-1-base
```

#### Datasets

Download the lambdalabs/pokemon-blip-captions  from [huggingface
page](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

```bash
mkdir -p data/datasets/pokemon-blip-captions
```

### Install Dependencies
Contact the Iluvatar administrator to get the missing packages:
  - deepspeed-0.16.4+corex.4.4.0.20250907-cp310-cp310-linux_x86_64.whl
  - triton-3.1.0+corex.4.4.0.20250907-cp310-cp310-linux_x86_64.whl
  - ixformer-0.6.0+corex.4.4.0.20250907-cp310-cp310-linux_x86_64.whl
  - flash_attn-2.6.3+corex.4.4.0.20250907-cp310-cp310-linux_x86_64.whl

```bash
# install packages
pip3 install http://files.deepspark.org.cn:880/deepspark/conformer/IXPyLogger-1.0.0-py3-none-any.whl
pip3 install huggingface_hub==0.25.1 transformers==4.38.1
pip3 install --upgrade pillow
pip3 install -r examples/text_to_image/requirements.txt
bash build_diffusers.sh && bash install_diffusers.sh
```

## Model Training

If you have downloaded the weights and dataset, please export the environment variables like below.

```bash
export CLIP_FLASH_ATTN=1
export USE_NHWC_GN=1
export USE_IXFORMER_GEGLU=0
export USE_APEX_LN=1
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
echo $ENABLE_FLASH_ATTENTION_WITH_IXDNN
cd examples/text_to_image
accelerate launch --config_file default_config.yaml --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=../../data/model_zoo/stabilityai/stable-diffusion-2-1-base \
  --dataset_name=../../data/datasets/pokemon-blip-captions \
  --resolution=512 \
  --seed 42 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-3" \
  --max_train_steps=100 \
  --NHWC \
  --dataloader_num_workers=32 \
  --apex_fused_adam "$@";

  exit ${EXIT_STATUS}
```

## Model Results

| Model  | GPUs    | ips_per_device | ips_per_gpu |
| ------ | ------- | -------------- | ----------- |
| SD 2.1 | BI-V150 x 16 | 6.65          | 13.3     |

## References

- [diffusers](https://github.com/huggingface/diffusers)
