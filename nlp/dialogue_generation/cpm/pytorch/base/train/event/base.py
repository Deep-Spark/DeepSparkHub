from typing import Tuple, List

import torch.nn
from torch import Tensor
from torch.optim import Optimizer
import utils

CPM_MODEL = torch.nn.Module
BatchType = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
NUM_GPUS  = int

class BaseTrainingEventInterface:

    def __init__(self, config):
        self.config = config

    def init_distributed_environment(self) -> Tuple[torch.device, NUM_GPUS]:
        return utils.init_dist_training_env(self.config)

    def device_synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def device_barrier(self):
        if torch.cuda.is_available():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
                torch.cuda.synchronize()

    def convert_model(self, model: CPM_MODEL) -> CPM_MODEL:
        return model

    def create_optimizer(self, model: CPM_MODEL) -> Optimizer:
        raise NotImplementedError()

    def model_to_fp16(self, model: CPM_MODEL, optimizer: Optimizer) -> Tuple[CPM_MODEL, Optimizer]:
        return model, optimizer

    def model_to_ddp(self, model: CPM_MODEL) -> CPM_MODEL:
        return model

    def on_init_start(self):
        pass

    def on_init_end(self):
        pass

    def on_backward(self, step: int, loss: Tensor, optimizer: Optimizer, grad_scaler=None):
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


