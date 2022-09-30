from typing import Tuple

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer

from optimizers import create_optimizer
from train.event.base import BaseTrainingEventInterface, SSD_MODEL
from train.training_state import TrainingState


class DefaultTrainingEvent(BaseTrainingEventInterface):

    def __init__(self, config):
        super(DefaultTrainingEvent, self).__init__(config)
        self.model = None
        self.optimizer = None
        self.autocast_ctx = None

    def save_checkpoint(self, path: str, training_state: TrainingState):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": training_state.epoch,
            "iter_num": training_state.iter_num,
        }, "{}/epoch{}_{}.pt".format(path, training_state.epoch, round(training_state.eval_ap, 5)))

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config.iteration = checkpoint["iter_num"]
        self.config.epoch = checkpoint["epoch"]

    def create_optimizer(self, model: SSD_MODEL) -> Optimizer:
        config = self.config
        current_momentum = 0.9
        current_lr = config.learning_rate * (config.train_batch_size * config.n_gpu / 32)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=current_lr,
                                momentum=current_momentum,
                                weight_decay=config.weight_decay_rate)
        return self.optimizer

    def model_to_fp16(self, model: SSD_MODEL, optimizer: Optimizer) -> Tuple[SSD_MODEL, Optimizer]:
        self.model = model
        self.optimizer = optimizer
        return self.model, self.optimizer

    def model_to_ddp(self, model: SSD_MODEL) -> SSD_MODEL:
        if self.config.distributed:
            self.model = NativeDDP(model,
                              device_ids=[self.config.local_rank])
        else:
            self.model = model
        return self.model

    def on_step_begin(self, step: int):
        self.autocast_ctx = autocast(self.config.fp16)
        self.autocast_ctx.__enter__()

    def on_backward(self, step: int, loss: torch.Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):
        self.autocast_ctx.__exit__(None, None, None)

        scaled_loss = grad_scaler.scale(loss)
        scaled_loss.backward()
        update_step = step % self.config.gradient_accumulation_steps == 0
        if update_step:
            self.update_model_params(optimizer, grad_scaler)

    def update_model_params(self, optimizer: Optimizer, grad_scaler: GradScaler=None):
        grad_scaler.step(optimizer)
        grad_scaler.update()

        for param in self.model.parameters():
            param.grad = None


