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
train_batch_size = 120
eval_batch_size = 160
learning_rate = 2.92e-3
weight_decay_rate = 1.6e-4
lr_decay_factor = 0.1
lr_decay_epochs = [44, 55]
warmup = 650
warmup_factor = 0
loss_scale = 0.0

# 3. Optimizer Configurations
num_workers = 4
fp16 = False
delay_allreduce = True


training_event = DefaultTrainingEvent