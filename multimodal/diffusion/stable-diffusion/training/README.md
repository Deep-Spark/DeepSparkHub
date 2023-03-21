# Stable Diffusion

## Model description
Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI, LAION and RunwayML. It's trained on 512x512 images from a subset of the LAION-5B database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 4GB VRAM. See the model card for more information.

## Prepare

### Setup env

```bash
pip3 install -r requirements.txt
```

###  Download

```bash
$ wget http://10.150.9.95/swapp/datasets/multimodal/stable_diffusion/pokemon-images.zip
$ unzip pokemon-images.zip
$ wget http://10.150.9.95/swapp/pretrained/multimodal/stable-diffusion/stable-diffusion-v1-4.zip
$ unzip stable-diffusion-v1-4.zip
```

## Train

### step 1   使用accelerate初始化训练环境

```bash
accelerate config  # 这里可以选择单卡或者多卡训练
                   # 这里建议只选择多卡或者单卡，其他优化选项例如：torch dynamo，deepspeed等均不建议使用

```

for example
single gpu

![image](IMG/single.png)

multi-gpu

![image](IMG/multi.png)

### step 2  开始训练

```bash
accelerate launch --mixed_precision="fp16" train_text_to_image.py  --pretrained_model_name_or_path=./stable-diffusion-v1-4  --use_ema  --resolution=512 --center_crop --random_flip  --train_batch_size=1  --gradient_accumulation_steps=4  --gradient_checkpointing  --max_train_steps=15000  --learning_rate=1e-05  --max_grad_norm=1  --lr_scheduler="constant" --lr_warmup_steps=0  --output_dir="sd-pokemon-model"  --caption_column 'additional_feature' --train_data_dir pokemon-images/datasets/images/train
```

## Test
```bash
python3 test.py
```
prompt:A pokemon with green eyes and red legs   

## Result
![image](IMG/pokemon.png)
