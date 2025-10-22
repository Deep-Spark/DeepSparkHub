source ../_utils/global_environment_variables.sh
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
export ENABLE_FLASH_ATTENTION_WITH_IXATTNBKD=0


: ${BATCH_SIZE:=8}

ROOT_DIR="$(cd "$(dirname "$0")/../.."; pwd)"
SRC_DIR=$ROOT_DIR/../models/multimodal/diffusion_model/stable-diffusion-2.1/pytorch/examples/text_to_image
DATASET_NAME=$ROOT_DIR/data/datasets/pokemon-blip-captions
MODEL_NAME=$ROOT_DIR/data/model_zoo/stabilityai/stable-diffusion-2-1
export DRT_MEMCPYUSEKERNEL=20000000000

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

# 训练用的一些环境变量，可以提高性能
export CLIP_FLASH_ATTN=1
export USE_NHWC_GN=1
export USE_IXFORMER_GEGLU=0
export USE_APEX_LN=1
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
echo $ENABLE_FLASH_ATTENTION_WITH_IXDNN


# 根据BI环境调整卡数
cd $SRC_DIR
actual_num_devices=$(ixsmi --list-gpus | wc -l)
num_devices=$(awk '/num_processes:/ {print $2}' $SRC_DIR/default_config.yaml)
echo $num_devices
echo $actual_num_devices
if [ "$num_devices" != "$actual_num_devices" ]; then
    echo "num_devices not matches actual_num_devices."
    sed -i "s/^num_processes:.*/num_processes: $actual_num_devices/" $SRC_DIR/default_config.yaml
fi

# 开始训练
accelerate launch --config_file default_config.yaml --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
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
  --apex_fused_adam "$@"; check_status

  exit ${EXIT_STATUS}
