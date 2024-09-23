# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn
from megatron_ds import get_args
# from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
# from transformer_engine.pytorch.module.rmsnorm import _RMSNorm
import ixformer.functions as F

if hasattr(F, "FusedRMSNorm"):
    use_ixformer = True
else:
    Warning("ixformer version is old. RMSNorm uses torch implementation in megatron-deepspeed")
    use_ixformer = False
class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = torch.Size((dim,))
        self.args = get_args()

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):

        ## apex implementation
        # return FusedRMSNormAffineMixedDtypesFunction.apply(x, self.weight, self.normalized_shape, self.eps)

        ## transformer_engine implementation
        # dtype = x.dtype
        # return _RMSNorm.apply(x, self.weight, self.eps, False, False, False, torch.is_grad_enabled(), dtype)

        ## ixformer implementation and torch implementation
        if use_ixformer and not self.args.RLHF:
            rmsn = F.FusedRMSNorm(self.normalized_shape, self.eps)
            rmsn.weight.data = self.weight
            return rmsn(x)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
