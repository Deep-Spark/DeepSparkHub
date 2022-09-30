from typing import Tuple, List

import torch.nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from train.training_state import TrainingState


SSD_MODEL = torch.nn.Module
BatchType = Tuple[Tensor, Tensor, Tensor]

class BaseTrainingEventInterface(object):

    def __init__(self, config):
        self.config = config

    def save_checkpoint(self, path: str, training_state: TrainingState):
        pass

    def load_checkpoint(self, checkpoint):
        pass

    def convert_model(self, model: SSD_MODEL) -> SSD_MODEL:
        return model

    def create_optimizer(self, model: SSD_MODEL) -> Optimizer:
        raise NotImplementedError()

    def model_to_fp16(self, model: SSD_MODEL, optimizer: Optimizer) -> Tuple[SSD_MODEL, Optimizer]:
        return model, optimizer

    def model_to_ddp(self, model: SSD_MODEL) -> SSD_MODEL:
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


