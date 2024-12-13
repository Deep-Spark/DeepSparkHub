: ${DATA:="/home/datasets/nlp/bert"}


SOCKETCORES=`lscpu|awk '/Core\(s\) per socket/ {print $4}'`

export INIT_LOSS_SCALE=1024
export GLOOGPU_BROADCAST=1
export GLOOGPU_ALLREDUCE=1
export GLOOGPU_REDUCESCATTER=1
export GLOOGPU_ALLGATHER=1
export GLOOGPU_MAXCHUNK=64


python3 -u -m bind_pyt \
--nproc_per_node auto --nsockets_per_node 2 \
--ncores_per_socket ${SOCKETCORES} --no_hyperthreads --no_membind \
run_pretraining.py --init_checkpoint ${DATA}/model.ckpt-28252.apex.pt \
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
--fp16 --fused_mha --distributed_lamb --dist_backend nccl \
--exchange_padding --target_mlm_accuracy 0.720 \
--seed=17103 --skip_checkpoint \
--train_batch_size 81 --learning_rate=3.6e-4 \
--max_steps 7400 --eval_batch_size 52 \
--gradient_accumulation_steps=3 "$@"