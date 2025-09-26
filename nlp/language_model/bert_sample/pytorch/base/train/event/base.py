# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from typing import Tuple, List

import torch.nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

BERT_MODEL = torch.nn.Module
BatchType =  Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

class BaseTrainingEventInterface:

    def __init__(self, config):
        self.config = config

    def convert_model(self, model: BERT_MODEL) -> BERT_MODEL:
        return model

    def create_optimizer(self, model: BERT_MODEL) -> Optimizer:
        raise NotImplementedError()

    def model_to_fp16(self, model: BERT_MODEL, optimizer: Optimizer) -> Tuple[BERT_MODEL, Optimizer]:
        return model, optimizer

    def model_to_ddp(self, model: BERT_MODEL) -> BERT_MODEL:
        return model

    def on_init_start(self):
        pass

    def on_init_end(self):
        pass

    def on_backward(self, step: int, loss: Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int):
        pass

    def on_step_begin(self, step: int):
        pass

    def on_step_end(self, step: int):
        pass


