import torch
from megatron.training.global_vars import get_args

from deepspeed.runtime.zero import GatheredParameters


def get_linear_layer(rows, columns, init_method, perform_initialization=True, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if perform_initialization:  # Take from modelparallel config
        with GatheredParameters(layer.weight, modifier_rank=0, enable=gather_params_on_init):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(layer.weight, modifier_rank=0, enable=gather_params_on_init):
            layer.bias.zero_()
    return layer
