from typing import ClassVar
from train.event.base import BaseTrainingEventInterface

# 1.Basic Configurations
# The train dir. Should contain coco datasets.
data_dir: str = None

# The backbone dir. The path to pretrained backbone weights file of resnet34-333f7ec4.pth, 'default is to get it from online torchvision repository'.
backbone_path: str = None

# Whether to run training.
do_train: bool = False

# Whether to read local rank from ENVVAR
use_env: bool = False

# device
device: str = None

# total gpu count
n_gpu: int = 1

distributed: bool = False

# local_rank for distributed training on gpus
local_rank: int = -1

# Communication backend for distributed training on gpus
dist_backend: str = "nccl"

# Stop training after reaching this Masked-LM accuracy
threshold: float = 0.23

# NMS threshold
nms_valid_thresh: float = 0.05

# Total number of training epochs to perform.
epochs: int = 80

# Start epoch, use for training from checkpoint
epoch: int = 0

# Start iteration, use for training from checkpoint
iteration: int = 0

# Sample to begin performing eval.
evaluation: list = [5, 10, 40, 50, 55, 60, 65, 70, 75, 80]

# Whether to save checkpoints
save_checkpoint: bool = False

# Where to save checkpoints
output: str = "./output"

# path to model checkpoint file
checkpoint: str = None

# random seed for initialization
seed: int = 42

# frequency of logging loss. If not positive, no logging is provided for training loss
log_freq: int = 20

# Number of updates steps to accumualte before performing a backward/update pass.
num_classes: int = 81

# Input images size
input_size: int = 300


# 2.Model Training Configurations
gradient_accumulation_steps: int = 1

# Total batch size for training.
train_batch_size: int = 120

# Total batch size for training.
eval_batch_size: int = 160

# The initial learning rate.
learning_rate: float = 2.92e-3

# weight decay rate.
weight_decay_rate: float = 1.6e-4

# decay rate of learning rate. default is 0.1.
lr_decay_factor: float = 0.1

# epochs at which to decay the learning rate.
lr_decay_epochs: list = [40, 50]

# How long the learning rate will be warmed up in fraction of epochs.
warmup: int = 650

# MLperf rule parameter for controlling warmup curve.
warmup_factor: int = 0

# Loss scaling, positive power of 2 values can improve fp16 convergence.
loss_scale: float = 0.0


# 3. Optimizer Configurations
# A object to provide some core components in training
training_event: ClassVar[BaseTrainingEventInterface] = None

# Dataloader workers
num_workers: int = 4

# Whether to use 16-bit float precision instead of 32-bit
fp16: bool = False

# Control training mode(FP32 or FP16) by opt_level using apex
# ToDo enum 0,1,2
opt_level: int = 0

delay_allreduce: bool = False

# Group of processes to collaborate on BatchNorm ops
# ToDo enum 1,2,4,8
bn_group: int = 1

fast_nms: bool = False

# Use fused color jitter
fast_cj: bool = False

#
use_coco_ext: bool = False

# Whether use Dali
dali: bool = False

# Run dali in synchronous mode instead of the (default) asynchronous
dali_sync: bool = False

# cache size (in GB) for Dali's nvjpeg caching
dali_cache: int = 0

# The following 4 optimization configurations must using Dali
nhwc: bool = False
pad_input: bool = False
jit: bool = False
use_nvjpeg: bool = False
