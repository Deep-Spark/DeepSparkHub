python3  -m torch.distributed.launch --nproc_per_node=8 --master_port 12333 \
  run_qa.py \
  --model_name_or_path "/home/data/bert/bert-base-uncased-pt" \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/