import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers import Cache
from transformers.utils import logging

from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer.linear import LinearWithFusedGradientAccu

try:
    from apex.corex.activations import SwiGLUFunction 
    swiglu_available=True
except:
    swiglu_available=False


logger = logging.get_logger(__name__)

class SwiGLU(torch.nn.Module):
    def forward(self, input):
        return SwiGLUFunction.apply(input)


class BaseLlamaMLP(LlamaMLP):
    """
    这个层主要的优化点是：将linear1(act(cat(linear2(x), linear3(x))))的结构变成 linear1(act(linear23(x)))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.gate_up = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.gate_up = LinearWithFusedGradientAccu(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down = LinearWithFusedGradientAccu(self.intermediate_size, self.hidden_size, bias=False)

        del self.gate_proj, self.up_proj, self.down_proj
        self.swiglu_available = swiglu_available
        if swiglu_available:
            self.act_fn = SwiGLU()

    def forward(self, x):
        if not self.swiglu_available:
            gate_proj, up_proj = self.gate_up(x).split((self.intermediate_size, self.intermediate_size), dim=-1)
            down_proj = self.down(self.act_fn(gate_proj) * up_proj)
        else:
            res = self.gate_up(x)
            down_proj = self.down(self.act_fn(res))
        return down_proj
        
        
class IXFLlamaMLP(BaseLlamaMLP):
    def __init__(self) -> None:
        raise NotImplementedError(
            "IXFLlamaMLP is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native LlamaAttention module to IXFLlamaMLP module provided above."
        )
        
    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        LazyInitContext.materialize(module)

        config = getattr(module, "config")
        
        mlp = BaseLlamaMLP(config=config)
        
        mlp.gate_up.weight.data = torch.concat((module.gate_proj.weight.data, module.up_proj.weight.data), dim=0)
        mlp.down.weight.data = module.down_proj.weight.data

        return mlp
