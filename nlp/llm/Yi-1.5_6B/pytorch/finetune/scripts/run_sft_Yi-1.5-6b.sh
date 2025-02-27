#/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/../sft/"

MODEL_PATH="/home/model_zoos/nlp/Yi-1.5-6B"
deepspeed main.py \
	--data_path ../yi_example_dataset/ \
	--model_name_or_path $MODEL_PATH \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--max_seq_len 4096 \
	--learning_rate 1e-6 \
	--weight_decay 0. \
	--num_train_epochs 2 \
	--training_debug_steps 40 \
	--gradient_accumulation_steps 1 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--output_dir ./finetuned_model \
    --print_loss
