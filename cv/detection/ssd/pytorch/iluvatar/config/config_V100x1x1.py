from training_event import ApexTrainingEvent

# 1.Basic Configurations
n_gpu = 1
distributed = False
dist_backend = "nccl"

save_checkpoint = False

seed = 1769250163
log_freq = 20


# 2.Model Training Configurations
gradient_accumulation_steps = 1
train_batch_size = 160
eval_batch_size = 160
learning_rate = 5.2e-3
weight_decay_rate = 1.6e-4
lr_decay_factor = 0.2
lr_decay_epochs = [34, 45]
warmup = 650
warmup_factor = 0
loss_scale = 0.0

# 3. Optimizer Configurations
num_workers = 4
fp16 = True
opt_level = 2
delay_allreduce = True
bn_group = 1
fast_nms = True
fast_cj = True
use_coco_ext = False
dali = True
dali_sync = False
dali_cache = -1
nhwc = True
pad_input = True
jit = True
use_nvjpeg = False


training_event = ApexTrainingEvent