mutable_params = [
    "dist_backend", "seed",
    "train_batch_size", "eval_batch_size", "learning_rate", "weight_decay_rate", "lr_decay_factor", "lr_decay_epochs",
    "warmup", "warmup_factor", "loss_scale",
    "gradient_accumulation_steps", "fp16", "opt_level", "delay_allreduce", "fast_nms", "fast_cj",
    "dali", "dali_cache", "nhwc", "pad_input", "jit", "use_nvjpeg",
    "training_event",
]

mutable_params += [
    "local_rank",
    "do_train",
    "data_dir",
    "backbone_path",
    "log_freq",
]

# only use for debug and fine tune.
mutable_params += [
    "save_checkpoint",
    "output",
    "checkpoint"
]