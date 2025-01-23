export WANDB_DISABLED=True
torchrun --standalone --nproc_per_node 1 train.py \
                      --model_id microsoft/Phi-3-mini-4k-instruct \
                      --dataset_name iamtarun/python_code_instructions_18k_alpaca \
                      --use_4bit \
                      --bnb_4bit_use_double_quant \
                      --output_dir phi-3-mini-4k-instruct-qlora-alpaca
