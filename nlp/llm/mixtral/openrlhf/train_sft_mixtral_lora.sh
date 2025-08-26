set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset Open-Orca/OpenOrca \
    --input_key question \
    --output_key response \
    --train_batch_size 16 \
    --micro_train_batch_size 1 \
    --max_samples 500000 \
    --pretrain mistralai/Mixtral-8x7B-v0.1 \
    --save_path ./checkpoint/mixtral-sft-lora\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn \
    --learning_rate 5e-6 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --aux_loss_coef 0.001 \
    --gradient_checkpointing_use_reentrant
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi