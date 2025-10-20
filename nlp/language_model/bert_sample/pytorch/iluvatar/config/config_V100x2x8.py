# /***************************************************************************************************
# * Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# * Copyright Declaration: This software, including all of its code and documentation,
# * except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# * Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# * Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# * CoreX. No user of this software shall have any right, ownership or interest in this software and
# * any use of this software shall be in compliance with the terms and conditions of the End User
# * License Agreement.
#  **************************************************************************************************/

from training_event import ApexTrainingEvent
from config_common import *
import os

fp16 = True
ddp_type = "apex"
dist_backend = "gloo"

gradient_accumulation_steps = 1
train_batch_size = 27
max_steps = 8000
start_warmup_step = 0
warmup_proportion = 0
warmup_steps = 0

# WARN: DistributedLAMB is not compatible with Gloo backend
# distributed_lamb = True
learning_rate = 4e-4
weight_decay_rate = 0.01
opt_lamb_beta_1 = 0.9
opt_lamb_beta_2 = 0.999

eval_batch_size = train_batch_size
max_samples_termination = 4500000
cache_eval_data = True

fused_gelu_bias = True
fused_mha = True
dense_seq_output = True
exchange_padding = True

dwu_num_rs_pg = 1
dwu_num_ar_pg = 1
dwu_num_blocks = 1

seed = 9031

target_mlm_accuracy = 0.71
save_checkpoint = True
log_freq = 200
eval_steps = 1000
init_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../../../../data/model_zoo/lm_bert/model.ckpt-0.pt')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../out')

training_event = ApexTrainingEvent