from training_event import DefaultTrainingEvent

# 1.Basic Configurations
n_gpu = 1
distributed = True
dist_backend = "nccl"

save_checkpoint = False

seed = 4230048668
log_freq = 20


# 2.Model Training Configurations
gradient_accumulation_steps = 1
train_batch_size = 128
eval_batch_size = 160
learning_rate = 2.5e-3
weight_decay_rate = 5e-4
lr_decay_factor = 0.1
lr_decay_epochs = [40, 50]
warmup = 300
warmup_factor = 0

# 3. Optimizer Configurations
num_workers = 4
fp16 = False
delay_allreduce = False


training_event = DefaultTrainingEvent