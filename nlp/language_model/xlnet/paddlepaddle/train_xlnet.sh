python3 -m paddle.distributed.launch --gpus "0" ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 500 \
    --output_dir ./tmp/
