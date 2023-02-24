export TASK_NAME=WNLI
python3  -m torch.distributed.launch --nproc_per_node=8 --master_port 12333 \
  run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir /tmp/$TASK_NAME/