import torch
from torch.autograd import Function
from apex import amp

from dlrm import cuda_ext


class DotBasedInteract(Function):
    """ Forward and Backward paths of cuda extension for dot-based feature interact."""

    @staticmethod
    @amp.half_function
    def forward(ctx, input, bottom_mlp_output, output_padding_width = 1):
        output = cuda_ext.dotBasedInteractFwd(input, bottom_mlp_output, output_padding_width)
        ctx.save_for_backward(input)
        ctx.output_padding_width = output_padding_width
        return output

    @staticmethod
    @amp.half_function
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output_padding_width = ctx.output_padding_width
        grad, mlp_grad = cuda_ext.dotBasedInteractBwd(input, grad_output, output_padding_width)
        return grad, mlp_grad, None


dotBasedInteract = DotBasedInteract.apply
