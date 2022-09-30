from typing import List, Union, Callable, Tuple

from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from .base import BaseTrainingEventInterface as TrainingEventInterface, SSD_MODEL
from .base_adapter import BaseTrainingEventMix, BaseTrainingEventAdapter
from train.training_state import TrainingState


class TrainingEventCompose(BaseTrainingEventAdapter):

    def __init__(self, interface: TrainingEventInterface, events: List[BaseTrainingEventAdapter]):
        super(TrainingEventCompose, self).__init__(interface.config)

        self.interface = interface
        self.events = events

    def launch(self):
        self._call_events_func(self.launch, with_interface=False)

    def save_checkpoint(self, path: str, training_state: TrainingState):
        self.interface.save_checkpoint(path, training_state)
        self._call_events_func(self.save_checkpoint, with_interface=False, path=path, training_state=training_state)

    def load_checkpoint(self, checkpoint):
        self.interface.load_checkpoint(checkpoint)
        self._call_events_func(self.load_checkpoint, with_interface=False, checkpoint=checkpoint)

    def convert_model(self, model: SSD_MODEL) -> SSD_MODEL:
        model = self.interface.convert_model(model)
        self._call_events_func(self.convert_model, with_interface=False, model=model)
        return model

    def create_optimizer(self, model: SSD_MODEL) -> Optimizer:
        optimizer = self.interface.create_optimizer(model)
        self._call_events_func(self.create_optimizer, with_interface=False, optimizer=optimizer)
        return optimizer

    def model_to_fp16(self, model: SSD_MODEL, optimizer: Optimizer) -> Tuple[SSD_MODEL, Optimizer]:
        model, optimizer = self.interface.model_to_fp16(model, optimizer)
        self._call_events_func(self.model_to_fp16, with_interface=False, model=model, optimizer=optimizer)
        return model, optimizer

    def model_to_ddp(self, model: SSD_MODEL) -> SSD_MODEL:
        model = self.interface.model_to_ddp(model)
        self._call_events_func(self.model_to_ddp, with_interface=False, model=model)
        return model

    def on_init_evaluate(self, result: dict):
        self._call_events_func(self.on_init_evaluate, with_interface=False, result=result)

    def on_evaluate(self, result: dict):
        self._call_events_func(self.on_evaluate, with_interface=False, result=result)

    def on_init_start(self):
        self._call_events_func(self.on_init_start, with_interface=True)

    def on_init_end(self):
        self._call_events_func(self.on_init_end, with_interface=True)

    def on_backward(self, step: int, loss: Tensor, optimizer: Optimizer, grad_scaler: GradScaler = None):
        return self.interface.on_backward(step, loss, optimizer, grad_scaler)

    def on_train_begin(self):
        self._call_events_func(self.on_train_begin, with_interface=True)

    def on_train_end(self):
        self._call_events_func(self.on_train_end, with_interface=True)

    def on_epoch_begin(self, epoch: int):
        self._call_events_func(self.on_epoch_begin, with_interface=True, epoch=epoch)

    def on_epoch_end(self, epoch: int):
        self._call_events_func(self.on_epoch_end, with_interface=True, epoch=epoch)

    def on_step_begin(self, step: int):
        self._call_events_func(self.on_step_begin, with_interface=True, step=step)

    def on_step_end(self, step: int, result: dict = None):
        self.interface.on_step_end(step)
        self._call_events_func(self.on_step_end, with_interface=False, step=step, result=result)

    def _call_events_func(self, func: Union[str, Callable], with_interface=False, *args, **kwargs):
        func_name = self._get_func_name(func)
        events = self.events
        if with_interface:
            events = [self.interface] + events

        result = []
        for event in events:
            ret = None
            if hasattr(event, func_name):
                 ret = getattr(event, func_name)(*args, **kwargs)
            result.append(ret)
        return result

    def _get_func_name(self, func: Union[str, Callable]):
        if isinstance(func, str):
            return func

        if callable(func):
            return func.__name__

        return None



