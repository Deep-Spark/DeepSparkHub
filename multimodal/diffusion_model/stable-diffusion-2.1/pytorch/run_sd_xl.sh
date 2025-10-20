export CLIP_FLASH_ATTN=1
export USE_NHWC_GN=1
export USE_IXFORMER_GEGLU=1
export USE_APEX_LN=1
export USE_NATIVE_ATTN=0
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
echo $ENABLE_FLASH_ATTENTION_WITH_IXDNN

export MODEL_NAME=/data/yili.li/jira_1040/stable-diffusion-xl-base-1.0
export DATASET_NAME=/data/yili.li/jira/pokemon-blip-captions/
export VAE_NAME=/data/yili.li/jira_1040/sdxl-vae-fp16-fix

cd /data/yili.li/jira_1068/diffusers/examples/text_to_image

accelerate launch --config_file zero2_config.yaml --mixed_precision="fp16"  train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --seed 42 \
  --gradient_checkpointing \
  --center_crop \
  --random_flip \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-3" \
  --max_train_steps=100 \
  --dataloader_num_workers=32 \
  --NHWC \
  --apex_fused_adam 
    # --use_ema 

