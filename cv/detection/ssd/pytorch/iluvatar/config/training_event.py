import os
from typing import Tuple

import torch
import apex
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from train.event.base import BaseTrainingEventInterface
from train.event.base import BatchType, SSD_MODEL
from train.training_state import TrainingState

from converter import convert_model


class ApexTrainingEvent(BaseTrainingEventInterface):

    def __init__(self, config):
        super(ApexTrainingEvent, self).__init__(config)
        self.model = None
        self.optimizer = None
        self.overflow_buf = None

    def save_checkpoint(self, path: str, training_state: TrainingState):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "amp": apex.amp.state_dict(),
            "master params": list(apex.amp.master_params(self.optimizer)),
            "epoch": training_state.epoch,
            "iter_num": training_state.iter_num,
        }, "{}/epoch{}_{}.pt".format(path, training_state.epoch, round(training_state.eval_ap, 5)))

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config.iteration = checkpoint["iter_num"]
        self.config.epoch = checkpoint["epoch"]
        if checkpoint.get("amp", None):
            apex.amp.load_state_dict(checkpoint["amp"])
        if checkpoint.get("master params", None):
            for param, saved_param in zip(apex.amp.master_params(self.optimizer), checkpoint["master params"]):
                param.data.copy_(saved_param.data)

    def on_init_start(self):
        pass

    def convert_model(self, model: SSD_MODEL) -> SSD_MODEL:
        self.model = convert_model(model, self.config)
        return self.model

    def create_optimizer(self, model: SSD_MODEL) -> Optimizer:
        config = self.config
        base_lr = 2.5e-3
        requested_lr_multiplier = config.learning_rate / base_lr
        adjusted_multiplier = max(1, round(requested_lr_multiplier * config.train_batch_size * config.n_gpu / 32))

        current_lr = base_lr * adjusted_multiplier
        current_weight_decay = config.weight_decay_rate

        self.optimizer = apex.optimizers.FusedSGD(model.parameters(),
                                         lr=current_lr,
                                         momentum=0.9,
                                         weight_decay=current_weight_decay)
        return self.optimizer

    def model_to_fp16(self, model: SSD_MODEL, optimizer: Optimizer) -> Tuple[SSD_MODEL, Optimizer]:
        self.model, self.optimizer = apex.amp.initialize(model, optimizer, opt_level="O{}".format(self.config.opt_level), loss_scale=128.)
        return self.model, self.optimizer

    def model_to_ddp(self, model: SSD_MODEL) -> SSD_MODEL:
        config = self.config
        if config.distributed:
            if config.delay_allreduce:
                print(config.local_rank, "Delaying allreduces to the end of backward()")
            self.model = apex.parallel.DistributedDataParallel(model,
                        gradient_predivide_factor=config.n_gpu / 8.0,
                        delay_allreduce=config.delay_allreduce,
                        retain_allreduce_buffers=config.fp16)
        else:
            self.model = model
        return self.model

    def on_step_begin(self, step: int):
        pass

    def on_step_end(self, step: int):
        pass

    def on_backward(self, step: int, loss: torch.Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        update_step = step % self.config.gradient_accumulation_steps == 0
        if update_step:
            self.update_model_params(optimizer, grad_scaler)

    def update_model_params(self, optimizer: Optimizer, grad_scaler: GradScaler=None):
        optimizer.step()
        for param in self.model.parameters():
            param.grad = None



