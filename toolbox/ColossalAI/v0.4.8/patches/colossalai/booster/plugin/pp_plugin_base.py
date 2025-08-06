from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

import torch

from colossalai.interface import ModelWrapper, OptimizerWrapper

from .plugin_base import Plugin


class PipelinePluginBase(Plugin):
    @abstractmethod
    def execute_pipeline(
        self,
        batch: Dict,
        model: ModelWrapper,
        criterion: Callable[[Any, Any], torch.Tensor],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        pass
