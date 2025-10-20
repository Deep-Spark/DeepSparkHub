: ${DATA:="/home/datasets/nlp/bert"}
export INIT_LOSS_SCALE=1024
export CUDA_VISIBLE_DEVICES=0
python3 run_pretraining_single.py --init_checkpoint ${DATA}/model.ckpt-28252.apex.pt \
--input_dir ${DATA}/2048_shards_uncompressed \
--bert_config_path ${DATA}/bert_config.json \
--eval_dir ${DATA}/eval_set_uncompressed \
--output_dir ./results/train_log_0818020305 \
--do_train --eval_iter_start_samples 150000 --eval_iter_samples 150000 \
--num_eval_examples 10000 --warmup_proportion=0 --warmup_steps=0 \
--start_warmup_step=0 --phase2 --max_seq_length=512 \
--max_predictions_per_seq=76 --train_mlm_accuracy_window_size=0 \
--max_samples_termination=4500000 --log_freq=10 --cache_eval_data \
--min_samples_to_start_checkpoints 150000 \
--num_samples_per_checkpoint 150000 \
--keep_n_most_recent_checkpoints 3 --opt_lamb_beta_1=0.9 \
--opt_lamb_beta_2=0.999 --weight_decay_rate=0.01 \
--fused_gelu_bias --dense_seq_output \
--dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-blocks=1 \
--fp16 --fused_mha  \
--exchange_padding --target_mlm_accuracy 0.720 \
--seed=17103 --skip_checkpoint \
--train_batch_size 72 --learning_rate=4.6e-4 \
--max_steps 40000 --eval_batch_size 24 \
--gradient_accumulation_steps=3 "$@"