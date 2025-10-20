
LR=1e-4

MASTER_ADDR="10.113.2.103"
MASTER_PORT=22233
HOSTFILE="hostfile"

deepspeed --num_nodes=2 --num_gpus=8 --master_addr=$MASTER_ADDR --master_port $MASTER_PORT --hostfile $HOSTFILE main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path "/home/model_zoo/nlp/chatglm-6b" \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate $LR \
    --fp16 \
    --deepspeed deepspeed.json
