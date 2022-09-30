import copy
import inspect
import os
import os.path as ospath
from typing import Tuple, Union, Iterable

from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from config.config_manager import get_properties_from_config
from utils.logging import PerfLogger, LogEvent, PerfLogLevel
from .base import SSD_MODEL
from .base_adapter import BaseTrainingEventAdapter


STACKLEVEL = 4


class TrainingLogger(BaseTrainingEventAdapter):

    def __init__(self, config, logger: PerfLogger=None, log_freq: int = 0):
        super(TrainingLogger, self).__init__(config)
        self.config = config
        self.log_freq = log_freq
        level = PerfLogLevel.INFO if log_freq > 0 else PerfLogLevel.SUBMITTION
        if logger is None:
            logger = PerfLogger.get_default_logger(rank=config.local_rank, level=level)
        self.logger = logger

        self.model = None
        self.submitter = None

    def launch(self):
        self.logger.log(LogEvent.launch_training, "Launch training", stacklevel=STACKLEVEL)
        config_path: str = self.config.config
        config_dict = get_properties_from_config(self.config)
        for key, value in config_dict.items():
            if type(value) not in [int, float, str, bool] and not isinstance(value, Iterable):
                config_dict[key] = str(value)

        # Extract definition of training event
        try:
            training_event_class = self.config.training_event
            if not inspect.isclass(training_event_class):
                training_event_class = training_event_class.__class__
            training_event_class_define = inspect.getabsfile(training_event_class)
            training_event_class_define = training_event_class_define.rsplit(".py", maxsplit=1)[0]
            training_event_class_define += ":" + training_event_class.__name__
        except:
            training_event_class_define = str(self.config.training_event)
        config_dict['training_event'] = training_event_class_define

        # Like /path/to/proj/submitter/model/config/config_xxx.py
        if config_path.startswith("."):
            config_path = ospath.abspath(config_path)

        config_path_nodes = config_path.rsplit(sep="/", maxsplit=4)
        submitter = config_path_nodes[1]
        model = config_path_nodes[2]
        self.logger.init_logger(submitter=submitter,
                                model=model,
                                config_path=config_path,
                                config=config_dict,
                                stacklevel=STACKLEVEL)

        self.model = model
        self.submitter = submitter

    def convert_model(self, model: SSD_MODEL):
        model_class = type(model)
        model_info = dict(
            type = model_class.__name__,
            module = model_class.__module__ if hasattr(model_class, "__module__") else "None"
        )
        self._log_event(LogEvent.convert_model, model_info)

    def create_optimizer(self, optimizer: Optimizer):
        optimizer_class = type(optimizer)
        optimizer_info = dict(
            type=optimizer_class.__name__,
            module=optimizer_class.__module__ if hasattr(optimizer_class, "__module__") else "None"
        )
        self._log_event(LogEvent.create_optimizer, optimizer_info)

    def model_to_fp16(self, model: SSD_MODEL, optimizer: Optimizer):
        fp16_info = dict(
            fp16 = self.config.fp16 if hasattr(self.config, "fp16") else False
        )
        self._log_event(LogEvent.model_to_fp16, fp16_info)

    def model_to_ddp(self, model: SSD_MODEL):
        model_class = type(model)
        model_info = dict(
            type=model_class.__name__,
            module=model_class.__module__ if hasattr(model_class, "__module__") else None
        )
        self._log_event(LogEvent.model_to_ddp, model_info)

    def on_init_evaluate(self, result: dict):
        self._log_event(LogEvent.init_evaluation, result)

    def on_evaluate(self, result: dict):
        self._log_event(LogEvent.evaluation, result)

    def on_init_start(self):
        self._log_event(LogEvent.init_start)

    def on_init_end(self):
        self._log_event(LogEvent.init_end, "Finish initialization")

    def on_backward(self, step: int, loss: Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):
        pass

    def on_train_begin(self):
        self._log_event(LogEvent.train_begin)

    def on_train_end(self):
        self._log_event(LogEvent.train_end)

    def on_epoch_begin(self, epoch: int):
        epoch_info = dict(epoch=epoch)
        self._log_event(LogEvent.epoch_begin, epoch_info)

    def on_epoch_end(self, epoch: int):
        epoch_info = dict(epoch=epoch)
        self._log_event(LogEvent.epoch_end, epoch_info)

    def on_step_begin(self, step: int):
        pass

    def on_step_end(self, step: int, result: dict=None):
        if (self.log_freq <= 0 or step % self.log_freq != 0) and step != 1:
            return
        if result is None:
            step_info = dict()
        else:
            step_info = copy.copy(result)

        step_info['step'] = step
        self._log_event(LogEvent.step_end, step_info)

    def _log_event(self, event, *args, **kwargs):
        self.logger.log(event, stacklevel=STACKLEVEL, *args, **kwargs)

