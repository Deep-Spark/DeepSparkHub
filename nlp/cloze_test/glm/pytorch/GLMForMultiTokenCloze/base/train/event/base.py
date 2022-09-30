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

    def model_to_fp16(self, model):
        return model

    def model_to_ddp(self, model: BERT_MODEL) -> BERT_MODEL:
        return model

    def on_init_start(self):
        pass

    def on_init_end(self):
        pass

    def on_backward(self,  setp, lm_loss, reduced_loss, optimizer, lr_scheduler):
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


