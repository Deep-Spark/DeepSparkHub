import torch
import torch.nn as nn

from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer.linear import LinearWithGradAccum

from ixformer.train import swiglu



class FusedMLP(torch.nn.Module):
    """
        Fused MLP with Gradient Accumulation Fusion
    """
    def __init__(self, hidden_size, intermediate_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w1 = LinearWithGradAccum(hidden_size, intermediate_size*2, bias=False)
        self.w2 = LinearWithGradAccum(intermediate_size, hidden_size, bias=False)

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:

        LazyInitContext.materialize(module)

        fused_mlp = FusedMLP(module.hidden_size, module.intermediate_size)
        fused_mlp.w1.weight.data = torch.concat((module.gate_proj.weight.data, module.up_proj.weight.data), dim=0)
        fused_mlp.w2.weight.data = module.down_proj.weight.data

        return fused_mlp

    def forward(self, hidden_states):
        current_hidden_states = self.w2(swiglu(self.w1(hidden_states)))
        return current_hidden_states
