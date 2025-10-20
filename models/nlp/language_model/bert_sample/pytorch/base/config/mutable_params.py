# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


mutable_params = [
    "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay_rate", "opt_lamb_beta_1",
    "opt_lamb_beta_2", "max_steps", "max_samples_termination", "warmup_proportion", "warmup_steps",
    "start_warmup_step", "dist_backend", "seed", "gradient_accumulation_steps", "fp16",
    "loss_scale", "exchange_padding", "enable_fuse_dropout", "disable_fuse_mask", "fused_gelu_bias",
    "fused_dropout_add", "dense_seq_output", "cache_eval_data", "training_event","output_dir","save_checkpoint","eval_steps","init_checkpoint","target_mlm_accuracy"
]

mutable_params += [
    "local_rank",
    "do_train",
    "data_dir",
    "log_freq"
]