"""Utilities for models."""

import math

import torch

from megatron.training import get_args
from megatron.legacy.model import LayerNorm, RMSNorm, RMSNormResidual
from megatron.core.jit import jit_fuser

from deepspeed.runtime.zero import GatheredParameters


def gather_and_init(param, init_method):
    with GatheredParameters(param, modifier_rank=0):
        init_method(param)

def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        attention_scores.masked_fill_(attention_mask_, -10000.0)
    else:
        attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
            layer.bias.zero_()
    return layer
