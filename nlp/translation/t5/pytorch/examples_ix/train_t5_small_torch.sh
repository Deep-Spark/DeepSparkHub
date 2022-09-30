EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=256 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_target_length=64 \
    --pad_to_max_length=True \
    --max_source_length=64 \
    --fp16=False \
    --fp16_backend=amp \
    --num_train_epochs 1 \
    --max_train_samples=900000  "$@";  check_status

exit ${EXIT_STATUS}