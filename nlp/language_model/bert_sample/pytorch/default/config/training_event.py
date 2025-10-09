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


from typing import Tuple

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer

from optimizers import create_optimizer
from train.event.base import BaseTrainingEventInterface, BERT_MODEL


class DefaultTrainingEvent(BaseTrainingEventInterface):

    def __init__(self, config):
        super(DefaultTrainingEvent, self).__init__(config)
        self.model = None
        self.optimizer = None
        self.num_iters_per_dataloader = 1

        self.autocast_ctx = None

    def create_optimizer(self, model: BERT_MODEL) -> Optimizer:
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        optimizer = create_optimizer('lamb', optimizer_grouped_parameters, self.config)

        self.model = model
        self.optimizer = optimizer
        return optimizer

    def model_to_fp16(self, model: BERT_MODEL, optimizer: Optimizer) -> Tuple[BERT_MODEL, Optimizer]:
        return model, optimizer

    def model_to_ddp(self, model: BERT_MODEL) -> BERT_MODEL:
        use_ddp = dist.is_initialized()
        if use_ddp:
            model = NativeDDP(model,
                              device_ids=[self.config.local_rank],
                              bucket_cap_mb=100,
                              gradient_as_bucket_view=self.config.use_gradient_as_bucket_view)
        self.model = model
        return model

    def on_step_begin(self, step: int):
        self.autocast_ctx = autocast(self.config.fp16)
        self.autocast_ctx.__enter__()

    def on_backward(self, step: int, loss: torch.Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):
        self.autocast_ctx.__exit__(None, None, None)

        scaled_loss = grad_scaler.scale(loss)
        scaled_loss.backward()
        update_step = step % self.config.gradient_accumulation_steps == 0
        if update_step:
            self.update_model_params(scaled_loss, optimizer, grad_scaler)

    def update_model_params(self, loss, optimizer: Optimizer, grad_scaler: GradScaler=None):
        grad_scaler.step(optimizer)
        grad_scaler.update()

        for param in self.model.parameters():
            param.grad = None


