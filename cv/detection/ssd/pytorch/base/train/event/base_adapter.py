from torch.optim import Optimizer

from .base import BaseTrainingEventInterface


class BaseTrainingEventMix:

    def launch(self):
        pass

    def create_optimizer(self, optimizer: Optimizer):
        pass

    def on_init_evaluate(self, result: dict):
        pass

    def on_evaluate(self, result: dict):
        pass

    def on_step_end(self, step: int, result: dict = None):
        pass


class BaseTrainingEventAdapter(BaseTrainingEventMix, BaseTrainingEventInterface):
    pass





