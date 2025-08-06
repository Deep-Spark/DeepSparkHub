import torch
from deepspeed.ops.op_builder import SwigluBuilder
global swiglu_cuda
class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swiglu_fwd(x)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return swiglu_bwd(x, g)

swiglu = SwiGLUFunction.apply
def swiglu_fwd(input: torch.Tensor):
    swiglu_cuda = SwigluBuilder().load()
    assert input.is_contiguous()
    assert input.dtype == torch.half or input.dtype == torch.float or input.dtype == torch.bfloat16
    assert input.size(-1) % 2 == 0
    return swiglu_cuda.swiglu_fwd(input)

def swiglu_bwd(input: torch.Tensor, grad: torch.Tensor):
    swiglu_cuda = SwigluBuilder().load()
    assert input.is_contiguous() and grad.is_contiguous()
    assert (input.dtype == torch.half and grad.dtype == torch.half) or (input.dtype == torch.float and grad.dtype == torch.float) or (input.dtype == torch.bfloat16 and grad.dtype == torch.bfloat16)
    assert input.size(-1) % 2 == 0 and input.size(-1) // grad.size(-1) == 2
    return swiglu_cuda.swiglu_bwd(input, grad)

