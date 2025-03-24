# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import io
import math
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter

from megatron.training.global_vars import get_args
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_and_expert_parallel_rank,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core import ixte_extensions

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, prepare_input_tensors_for_wgrad_compute
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.utils import VocabUtility, divide, split_tensor_along_last_dim
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _grad_accum_fusion_available,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
    linear_with_frozen_weight,
    LinearWithGradAccumulationAndAsyncCommunication,
)

try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    pass


class SequenceParallelPositionEmbedding(torch.nn.Module):
    """Embedding parallelized in the sequence dimension.

    Arguments:
        sequence_length: max sequence length.
        embedding_dim: size of hidden state.
    """

    def __init__(self, sequence_length, embedding_dim):
        super(SequenceParallelPositionEmbedding, self).__init__()
        sequence_parallel_size = get_tensor_model_parallel_world_size()
        assert sequence_length % sequence_parallel_size == 0
        local_sequence_length = sequence_length // sequence_parallel_size
        self.offset = local_sequence_length * get_tensor_model_parallel_rank()
        self.local_embeddings = torch.nn.Embedding(
            local_sequence_length, embedding_dim)

    def forward(self, position_ids):
        return self.local_embeddings(position_ids - self.offset)

def gradientUpdateFunction(total_input, grad_output, weight):
    if weight.grad == None:
        weight.grad = grad_output.t().matmul(total_input)
    else:
        weight.grad += grad_output.t().matmul(total_input)

def linear_with_grad_accumulation_and_async_allreduce_forward(
    ctx,
    input,
    weight,
    bias,
    gradient_accumulation_fusion,
    allreduce_dgrad,
    sequence_parallel,
    grad_output_buffer,
    wgrad_deferral_limit,
    inference_params=None,
):
    ctx.save_for_backward(input, weight)
    ctx.use_bias = bias is not None
    ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.sequence_parallel = sequence_parallel
    ctx.wgrad_deferral_limit = wgrad_deferral_limit
    ctx.grad_output_buffer = grad_output_buffer

    if sequence_parallel and not inference_params:
        world_size = get_tensor_model_parallel_world_size()
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        torch.distributed._all_gather_base(
            all_gather_buffer, input, group=get_tensor_model_parallel_group()
        )
        total_input = all_gather_buffer
    else:
        total_input = input

    output = torch.matmul(total_input, weight.t())
    if bias is not None:
        output = output + bias
    return output

def linear_with_grad_accumulation_and_async_allreduce_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    use_bias = ctx.use_bias
    grad_output_buffer = ctx.grad_output_buffer
    wgrad_deferral_limit = ctx.wgrad_deferral_limit

    wgrad_compute = True
    if grad_output_buffer is not None:
        if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

    if wgrad_compute:
        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel and wgrad_compute:
        handle.wait()

    if wgrad_compute:
        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if grad_output.dim() == 3:
            grad_output = grad_output.view(
                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            )
            total_input = total_input.view(
                total_input.shape[0] * total_input.shape[1], total_input.shape[2]
            )
        else:
            # Somehow when DeepSpeed MoE is used, grad_output could have 4 dimensions.
            # TODO: May need further investigation
            total_input = total_input.contiguous()
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

    if ctx.allreduce_dgrad:
        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(
            grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # all-reduce is scheduled before the weight gradient computation

    if ctx.sequence_parallel:
        assert not ctx.allreduce_dgrad
        dim_size = list(input.size())
        sub_grad_input = torch.empty(
            dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        # reduce_scatter
        handle = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # reduce scatter is scheduled before the weight gradient computation

    if ctx.gradient_accumulation_fusion:
        if wgrad_compute:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

        if hasattr(weight, 'grad_added_to_main_grad'):
            # When overlap_grad_reduce is True, need to ensure that backward hooks
            # are all run on the main backprop thread to prevent deadlocks. Setup
            # dummy grad_weight tensor to prevent backward hooks from being run
            # in a background thread.
            if getattr(weight, 'zero_out_wgrad', False):
                grad_weight = torch.zeros(
                    weight.main_grad.shape,
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                grad_weight = torch.empty(
                    weight.main_grad.shape,
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            weight.grad_added_to_main_grad = True
        else:
            grad_weight = None
    else:
        grad_weight = grad_output.t().matmul(total_input)
    # from megatronspeed.core.tensor_parallel.weight_grad_store import WeightGradStore
    # WeightGradStore.put(total_input, grad_output, weight, gradientUpdateFunction)
    # grad_weight = None
    grad_bias = grad_output.sum(dim=0) if use_bias else None

    if ctx.sequence_parallel:
        handle.wait()
        # Need to return None's as gradient has to flow for all the input arguments
        # provided during forward
        return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None

    if ctx.allreduce_dgrad:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

# class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
#     """See linear_with_grad_accumulation_and_async_allreduce"""

#     @staticmethod
#     @custom_fwd
#     def forward(
#         ctx,
#         input,
#         weight,
#         bias,
#         gradient_accumulation_fusion,
#         allreduce_dgrad,
#         sequence_parallel,
#         grad_output_buffer,
#         wgrad_deferral_limit,
#     ):
#         ctx.save_for_backward(input, weight)
#         ctx.use_bias = bias is not None
#         ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
#         ctx.allreduce_dgrad = allreduce_dgrad
#         ctx.sequence_parallel = sequence_parallel
#         ctx.wgrad_deferral_limit = wgrad_deferral_limit
#         ctx.grad_output_buffer = grad_output_buffer

#         if sequence_parallel:
#             world_size = get_tensor_model_parallel_world_size()
#             dim_size = list(input.size())
#             dim_size[0] = dim_size[0] * world_size

#             all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
#             torch.distributed._all_gather_base(
#                 all_gather_buffer, input, group=get_tensor_model_parallel_group()
#             )
#             total_input = all_gather_buffer
#         else:
#             total_input = input

#         output = torch.matmul(total_input, weight.t())
#         if bias is not None:
#             output = output + bias
#         return output

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         use_bias = ctx.use_bias
#         grad_output_buffer = ctx.grad_output_buffer
#         wgrad_deferral_limit = ctx.wgrad_deferral_limit

#         wgrad_compute = True
#         if grad_output_buffer is not None:
#             if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
#                 grad_output_buffer.append(grad_output)
#                 wgrad_compute = False

#         if wgrad_compute:
#             if ctx.sequence_parallel:
#                 world_size = get_tensor_model_parallel_world_size()
#                 dim_size = list(input.size())
#                 dim_size[0] = dim_size[0] * world_size

#                 all_gather_buffer = get_global_memory_buffer().get_tensor(
#                     dim_size, input.dtype, "mpu"
#                 )
#                 handle = torch.distributed._all_gather_base(
#                     all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
#                 )

#                 # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#                 # gather is scheduled before the input gradient computation
#                 total_input = all_gather_buffer
#             else:
#                 total_input = input
#         grad_input = grad_output.matmul(weight)

#         if ctx.sequence_parallel and wgrad_compute:
#             handle.wait()

#         if wgrad_compute:
#             # Doing gather + slicing during the NeMo forward pass can make this tensor
#             # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
#             # clones it if it's not contiguous:
#             # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
#             grad_output = grad_output.contiguous()
#             # Convert the tensor shapes to 2D for execution compatibility
#             if grad_output.dim() == 3:
#                 grad_output = grad_output.view(
#                     grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
#                 )
#                 total_input = total_input.view(
#                     total_input.shape[0] * total_input.shape[1], total_input.shape[2]
#                 )
#             else:
#                 # Somehow when DeepSpeed MoE is used, grad_output could have 4 dimensions.
#                 # TODO: May need further investigation
#                 total_input = total_input.contiguous()
#                 grad_output = grad_output.view(-1, grad_output.shape[-1])
#                 total_input = total_input.view(-1, total_input.shape[-1])

#         if ctx.allreduce_dgrad:
#             # Asynchronous all-reduce
#             handle = torch.distributed.all_reduce(
#                 grad_input, group=get_tensor_model_parallel_group(), async_op=True
#             )
#             # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#             # all-reduce is scheduled before the weight gradient computation

#         if ctx.sequence_parallel:
#             assert not ctx.allreduce_dgrad
#             dim_size = list(input.size())
#             sub_grad_input = torch.empty(
#                 dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
#             )
#             # reduce_scatter
#             handle = torch.distributed._reduce_scatter_base(
#                 sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
#             )
#             # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#             # reduce scatter is scheduled before the weight gradient computation

#         if ctx.gradient_accumulation_fusion:
#             if wgrad_compute:
#                 if weight.main_grad.dtype == torch.float32:
#                     fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
#                         total_input, grad_output, weight.main_grad
#                     )
#                 elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
#                     fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
#                         total_input, grad_output, weight.main_grad
#                     )
#                 else:
#                     raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

#             if hasattr(weight, 'grad_added_to_main_grad'):
#                 # When overlap_grad_reduce is True, need to ensure that backward hooks
#                 # are all run on the main backprop thread to prevent deadlocks. Setup
#                 # dummy grad_weight tensor to prevent backward hooks from being run
#                 # in a background thread.
#                 if getattr(weight, 'zero_out_wgrad', False):
#                     grad_weight = torch.zeros(
#                         weight.main_grad.shape,
#                         dtype=input.dtype,
#                         device=torch.cuda.current_device(),
#                         requires_grad=False,
#                     )
#                 else:
#                     grad_weight = torch.empty(
#                         weight.main_grad.shape,
#                         dtype=input.dtype,
#                         device=torch.cuda.current_device(),
#                         requires_grad=False,
#                     )
#                 weight.grad_added_to_main_grad = True
#             else:
#                 grad_weight = None
#         else:
#             grad_weight = grad_output.t().matmul(total_input)
#         from megatronspeed.core.tensor_parallel.weight_grad_store import WeightGradStore
#         WeightGradStore.put(total_input, grad_output, weight, gradientUpdateFunction)
#         grad_bias = grad_output.sum(dim=0) if use_bias else None

#         if ctx.sequence_parallel:
#             handle.wait()
#             # Need to return None's as gradient has to flow for all the input arguments
#             # provided during forward
#             return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

#         if ctx.allreduce_dgrad:
#             handle.wait()

#         return grad_input, grad_weight, grad_bias, None, None, None, None, None

def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    allreduce_dgrad: bool = None,
    inference_params = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."


        async_grad_allreduce (bool required): Do the allreduce of input
            gradients asyncronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.


        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Defaults to 0.

        allreduce_dgrad (bool): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.
    """
    if allreduce_dgrad is None:
        warnings.warn(
            "async_grad_allreduce is deprecated and will be removed in a future release. use allreduce_dgrad instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        inference_params,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)

linear_with_grad_accumulation_and_async_allreduce.warned = False

def column_parallel_linear_init(self,
    input_size,
    output_size,
    *,
    config: ModelParallelConfig,
    init_method: Callable,
    bias=True,
    gather_output=False,
    stride=1,
    keep_master_weight_for_test=False,
    skip_bias_add=False,
    skip_weight_param_allocation: bool = False,
    embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    is_expert: bool = False,
    tp_comm_buffer_name: str = None,  # Not used
    disable_grad_reduce: bool = False,
    is_logits_gemm: bool = False,
    moe=False, enable_expert_tensor_parallelism=False,
):
    super(ColumnParallelLinear, self).__init__()

    # Keep input parameters
    self.input_size = input_size
    self.output_size = output_size
    self.gather_output = gather_output
    # Divide the weight matrix along the last dimension.
    self.skip_bias_add = skip_bias_add
    self.is_expert = is_expert
    self.expert_parallel = config.expert_model_parallel_size > 1
    self.embedding_activation_buffer = embedding_activation_buffer
    self.grad_output_buffer = grad_output_buffer
    self.config = config
    self.disable_grad_reduce = disable_grad_reduce

    args = get_args()
    self.deepspeed = args.deepspeed
    self.explicit_expert_comm = False
    rank = get_tensor_model_parallel_rank()
    if not args.deepspeed:
        self.explicit_expert_comm = self.is_expert and (
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
    else:
        if moe and (not enable_expert_tensor_parallelism):
            world_size = 1
            self.is_expert_without_slicing = True
        else:
            world_size = get_tensor_model_parallel_world_size()
            self.is_expert_without_slicing = False

    self.output_size_per_partition = divide(output_size, world_size)

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # Initialize weight.
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )

        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
    else:
        self.weight = None

    if bias:
        if config.use_cpu_initialization:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
            )
        else:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
        set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
        if config.perform_initialization:
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
    else:
        self.register_parameter('bias', None)

    self.sequence_parallel = config.sequence_parallel
    if self.sequence_parallel and world_size <= 1:
        warnings.warn(
            f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
            f"Disabling sequence parallel."
        )
        self.sequence_parallel = False

    self.allreduce_dgrad = (
        world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
    )

    if args.deepspeed:
        self.allreduce_dgrad = False

    if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
        raise RuntimeError(
            "ColumnParallelLinear was called with gradient_accumulation_fusion set "
            "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
            "module is not found. To use gradient_accumulation_fusion you must "
            "install APEX with --cpp_ext and --cuda_ext. For example: "
            "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
            "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
            "gradient accumulation fusion."
        )
    self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

    if self.allreduce_dgrad and self.sequence_parallel:
        raise RuntimeError(
            "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
        )

    self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    # Hook adding a default empty _extra_state for state dict
    self._register_load_state_dict_pre_hook(
        lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
            f'{prefix}_extra_state'
        )
    )
    self.use_ixte = False
    if is_logits_gemm and config.sequence_parallel and ixte_extensions._USE_IXTE and config.transformer_impl == "transformer_engine":
        self.use_ixte = True

def column_parallel_linear_forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, inference_params=None):
    """Forward of ColumnParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        weight (optional): weight tensor to use, compulsory when
            skip_weight_param_allocation is True.

    Returns:
        - output
        - bias

    """
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # Check the weight passed in is the correct shape
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    bias = self.bias if not self.skip_bias_add else None

    if self.use_ixte:
        output = ixte_extensions.get_logits_linear_func()(
            input=input_,
            weight=weight,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(),
        )
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    if (
        self.allreduce_dgrad
        or self.sequence_parallel
        or self.explicit_expert_comm
        or self.disable_grad_reduce
        or (self.deepspeed and self.is_expert_without_slicing)
    ):
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)

    if self.config.defer_embedding_wgrad_compute:
        if (
            self.config.wgrad_deferral_limit == 0
            or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
        ):
            self.embedding_activation_buffer.append(input_parallel)

    # Matrix multiply.
    if not weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=allreduce_dgrad,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        grad_output_buffer=(
            self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
        ),
        wgrad_deferral_limit=(
            self.config.wgrad_deferral_limit
            if self.config.defer_embedding_wgrad_compute
            else None
        ),
        allreduce_dgrad=allreduce_dgrad,
        inference_params=inference_params,
    )
    if (self.gather_output and not self.deepspeed) or \
        (self.deepspeed and self.gather_output and not self.is_expert_without_slicing):
        # All-gather across the partitions.
        assert not self.sequence_parallel
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias

# class ColumnParallelLinear(torch.nn.Module):
#     """Linear layer with column parallelism.

#     The linear layer is defined as Y = XA + b. A is parallelized along
#     its second dimension as A = [A_1, ..., A_p].

#     Args:
#         input_size: first dimension of matrix A.
#         output_size: second dimension of matrix A.
#         bias: If true, add bias
#         gather_output: If true, call all-gather on output and make Y available to all GPUs, otherwise, every GPU will have its output which is Y_i = XA_i
#         init_method: method to initialize weights. Note that bias is always set to zero.
#         stride: For the strided linear layers.
#         keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
#         skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
#         skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed as a keyword argument `weight` during the forward pass. Note that this does not affect bias, which will be allocated if bias is True. Defaults to False.
#         embedding_activation_buffer: This buffer holds the input activations of the final embedding linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
#         grad_output_buffer: This buffer holds the gradient outputs of the final embedding linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
#         is_expert: If True, the layer is treated as an MoE expert layer.
#         config: ModelParallelConfig object
#         tp_comm_buffer_name: Communication buffer name is not used in non-Transformer-Engine modules.
#         disable_grad_reduce: If True, reduction of output gradients across tensor-parallel ranks will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to delay and fuse reduction along with other gradients for performance optimization.
#     """

#     def __init__(
#         self,
#         input_size,
#         output_size,
#         *,
#         config: ModelParallelConfig,
#         init_method: Callable,
#         bias=True,
#         gather_output=False,
#         stride=1,
#         keep_master_weight_for_test=False,
#         skip_bias_add=False,
#         skip_weight_param_allocation: bool = False,
#         embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
#         grad_output_buffer: Optional[List[torch.Tensor]] = None,
#         is_expert: bool = False,
#         tp_comm_buffer_name: str = None,  # Not used
#         disable_grad_reduce: bool = False,
#         is_logits_gemm: bool = False,
#         moe=False, enable_expert_tensor_parallelism=False,
#     ):
#         super(ColumnParallelLinear, self).__init__()

#         # Keep input parameters
#         self.input_size = input_size
#         self.output_size = output_size
#         self.gather_output = gather_output
#         # Divide the weight matrix along the last dimension.
#         self.skip_bias_add = skip_bias_add
#         self.is_expert = is_expert
#         self.expert_parallel = config.expert_model_parallel_size > 1
#         self.embedding_activation_buffer = embedding_activation_buffer
#         self.grad_output_buffer = grad_output_buffer
#         self.config = config
#         self.disable_grad_reduce = disable_grad_reduce

#         args = get_args()
#         self.deepspeed = args.deepspeed
#         self.explicit_expert_comm = False
#         rank = get_tensor_model_parallel_rank()
#         if not args.deepspeed:
#             self.explicit_expert_comm = self.is_expert and (
#                 config.tensor_model_parallel_size > 1 or self.expert_parallel
#             )
#             if self.explicit_expert_comm and config.moe_extended_tp:
#                 world_size = get_tensor_and_expert_parallel_world_size()
#                 rank = get_tensor_and_expert_parallel_rank()
#             else:
#                 world_size = get_tensor_model_parallel_world_size()
#                 rank = get_tensor_model_parallel_rank()
#         else:
#             if moe and (not enable_expert_tensor_parallelism):
#                 world_size = 1
#                 self.is_expert_without_slicing = True
#             else:
#                 world_size = get_tensor_model_parallel_world_size()
#                 self.is_expert_without_slicing = False

#         self.output_size_per_partition = divide(output_size, world_size)

#         # Parameters.
#         # Note: torch.nn.functional.linear performs XA^T + b and as a result
#         # we allocate the transpose.
#         # Initialize weight.
#         if not skip_weight_param_allocation:
#             if config.use_cpu_initialization:
#                 self.weight = Parameter(
#                     torch.empty(
#                         self.output_size_per_partition, self.input_size, dtype=config.params_dtype
#                     )
#                 )
#                 if config.perform_initialization:
#                     self.master_weight = _initialize_affine_weight_cpu(
#                         self.weight,
#                         self.output_size,
#                         self.input_size,
#                         self.output_size_per_partition,
#                         0,
#                         init_method,
#                         stride=stride,
#                         return_master_weight=keep_master_weight_for_test,
#                         rank=rank,
#                         world_size=world_size,
#                     )
#             else:
#                 self.weight = Parameter(
#                     torch.empty(
#                         self.output_size_per_partition,
#                         self.input_size,
#                         device=torch.cuda.current_device(),
#                         dtype=config.params_dtype,
#                     )
#                 )
#                 if config.perform_initialization:
#                     _initialize_affine_weight_gpu(
#                         self.weight,
#                         init_method,
#                         partition_dim=0,
#                         stride=stride,
#                         expert_parallel=(self.is_expert and self.expert_parallel),
#                     )

#             setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
#         else:
#             self.weight = None

#         if bias:
#             if config.use_cpu_initialization:
#                 self.bias = Parameter(
#                     torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
#                 )
#             else:
#                 self.bias = Parameter(
#                     torch.empty(
#                         self.output_size_per_partition,
#                         device=torch.cuda.current_device(),
#                         dtype=config.params_dtype,
#                     )
#                 )
#             set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
#             if config.perform_initialization:
#                 # Always initialize bias to zero.
#                 with torch.no_grad():
#                     self.bias.zero_()
#             setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
#         else:
#             self.register_parameter('bias', None)

#         self.sequence_parallel = config.sequence_parallel
#         if self.sequence_parallel and world_size <= 1:
#             warnings.warn(
#                 f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
#                 f"Disabling sequence parallel."
#             )
#             self.sequence_parallel = False

#         self.allreduce_dgrad = (
#             world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
#         )

#         if args.deepspeed:
#             self.allreduce_dgrad = False

#         if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
#             raise RuntimeError(
#                 "ColumnParallelLinear was called with gradient_accumulation_fusion set "
#                 "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
#                 "module is not found. To use gradient_accumulation_fusion you must "
#                 "install APEX with --cpp_ext and --cuda_ext. For example: "
#                 "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
#                 "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
#                 "gradient accumulation fusion."
#             )
#         self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

#         if self.allreduce_dgrad and self.sequence_parallel:
#             raise RuntimeError(
#                 "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
#             )

#         self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

#         # Hook adding a default empty _extra_state for state dict
#         self._register_load_state_dict_pre_hook(
#             lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
#                 f'{prefix}_extra_state'
#             )
#         )
#         self.use_ixte = False
#         if is_logits_gemm and config.sequence_parallel and ixte_extensions._USE_IXTE and config.transformer_impl == "transformer_engine":
#             self.use_ixte = True

#     def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, recompute_fwd = False):
#         """Forward of ColumnParallelLinear

#         Args:
#             input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

#             weight (optional): weight tensor to use, compulsory when
#                 skip_weight_param_allocation is True.

#         Returns:
#             - output
#             - bias

#         """
#         if weight is None:
#             if self.weight is None:
#                 raise RuntimeError(
#                     "weight was not supplied to ColumnParallelLinear forward pass "
#                     "and skip_weight_param_allocation is True."
#                 )
#             weight = self.weight
#         else:
#             # Check the weight passed in is the correct shape
#             expected_shape = (self.output_size_per_partition, self.input_size)
#             if weight.shape != expected_shape:
#                 raise RuntimeError(
#                     f"supplied weight's shape is {tuple(weight.shape)}, "
#                     f"not {expected_shape} as expected"
#                 )

#         if self.config._cpu_offloading_context is not None:
#             if self.config._cpu_offloading_context.inside_context == True:
#                 assert (
#                     self.config.cpu_offloading == False
#                 ), "CPU Offloading cannot be enabled while using non-TE modules"

#         bias = self.bias if not self.skip_bias_add else None

#         if self.use_ixte:
#             output = ixte_extensions.get_logits_linear_func()(
#                 input=input_,
#                 weight=weight,
#                 sequence_parallel=self.sequence_parallel,
#                 gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#                 tp_group=get_tensor_model_parallel_group(),
#             )
#             output_bias = self.bias if self.skip_bias_add else None
#             return output, output_bias
#         if (
#             self.allreduce_dgrad
#             or self.sequence_parallel
#             or self.explicit_expert_comm
#             or self.disable_grad_reduce
#             or (self.deepspeed and self.is_expert_without_slicing)
#         ):
#             input_parallel = input_
#         else:
#             input_parallel = copy_to_tensor_model_parallel_region(input_)

#         if self.config.defer_embedding_wgrad_compute:
#             if (
#                 self.config.wgrad_deferral_limit == 0
#                 or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
#             ):
#                 self.embedding_activation_buffer.append(input_parallel)

#         # Matrix multiply.
#         if not weight.requires_grad:
#             self._forward_impl = linear_with_frozen_weight
#         else:
#             self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

#         allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

#         output_parallel = self._forward_impl(
#             input=input_parallel,
#             weight=weight,
#             bias=bias,
#             gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#             async_grad_allreduce=allreduce_dgrad,
#             sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
#             grad_output_buffer=(
#                 self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
#             ),
#             wgrad_deferral_limit=(
#                 self.config.wgrad_deferral_limit
#                 if self.config.defer_embedding_wgrad_compute
#                 else None
#             ),
#             allreduce_dgrad=allreduce_dgrad,
#         )
#         if (self.gather_output and not self.deepspeed) or \
#             (self.deepspeed and self.gather_output and not self.is_expert_without_slicing):
#             # All-gather across the partitions.
#             assert not self.sequence_parallel
#             output = gather_from_tensor_model_parallel_region(output_parallel)
#         else:
#             output = output_parallel
#         output_bias = self.bias if self.skip_bias_add else None
#         return output, output_bias

#     def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
#         """Sharding along axis 0, bias sharded"""
#         state_dict = self.state_dict(prefix='', keep_vars=True)
#         return make_sharded_tensors_for_checkpoint(
#             state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
#         )

#     def set_extra_state(self, state: Any):
#         """Extra state is ignored"""

#     def get_extra_state(self) -> None:
#         """Keep compatibility with TE state dict."""
#         return None

def row_parallel_linear_init(
    self,
    input_size: int,
    output_size: int,
    *,
    config: ModelParallelConfig,
    init_method: Callable,
    bias: bool,
    input_is_parallel: bool,
    skip_bias_add: bool,
    stride: int = 1,
    keep_master_weight_for_test: bool = False,
    is_expert: bool = False,
    tp_comm_buffer_name: str = None,  # Not used
    moe=False, enable_expert_tensor_parallelism=False,
):
    super(RowParallelLinear, self).__init__()

    # Keep input parameters
    self.input_size = input_size
    self.output_size = output_size
    self.input_is_parallel = input_is_parallel
    self.skip_bias_add = skip_bias_add
    self.config = config
    self.is_expert = is_expert
    self.expert_parallel = config.expert_model_parallel_size > 1
    self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
    self.sequence_parallel = config.sequence_parallel
    if self.sequence_parallel and not self.input_is_parallel:
        raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

    args = get_args()
    self.deepspeed = args.deepspeed
    self.explicit_expert_comm = False
    rank = get_tensor_model_parallel_rank()
    if not args.deepspeed:
        self.explicit_expert_comm = self.is_expert and (
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:
            world_size = get_tensor_and_expert_parallel_world_size()
            rank = get_tensor_and_expert_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
    else:
        if moe and (not enable_expert_tensor_parallelism):
            world_size = 1
        else:
            world_size = get_tensor_model_parallel_world_size()
        self.is_expert_without_slicing = moe and world_size==1

    self.input_size_per_partition = divide(input_size, world_size)

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # Initialize weight.
    if config.use_cpu_initialization:
        self.weight = Parameter(
            torch.empty(
                self.output_size, self.input_size_per_partition, dtype=config.params_dtype
            )
        )
        if config.perform_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
            )
    else:
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.weight,
                init_method,
                partition_dim=1,
                stride=stride,
                expert_parallel=(self.is_expert and self.expert_parallel),
            )
    setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

    if bias:
        if config.use_cpu_initialization:
            self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
        else:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )

        if config.perform_initialization:
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
    else:
        self.register_parameter('bias', None)

    self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    # Hook adding a default empty _extra_state for state dict
    self._register_load_state_dict_pre_hook(
        lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
            f'{prefix}_extra_state'
        )
    )

def row_parallel_linear_forward(self, input_, inference_params=None):
    """Forward of RowParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

    Returns:
        - output
        - bias
    """

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    # Set up backprop all-reduce.
    if self.input_is_parallel or (self.deepspeed and self.is_expert_without_slicing):
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    allreduce_dgrad = False

    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=allreduce_dgrad,
        sequence_parallel=False,
        grad_output_buffer=None,
        allreduce_dgrad=allreduce_dgrad,
        inference_params=inference_params,
    )

    # All-reduce across all the partitions.
    if self.explicit_expert_comm:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.deepspeed and self.is_expert_without_slicing: # non-expert only tensor-parallelism
        output_ = output_parallel
    elif self.sequence_parallel and not inference_params:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias

# class RowParallelLinear(torch.nn.Module):
#     """Linear layer with row parallelism.

#     The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

#     Args:
#         input_size: first dimension of matrix A.
#         output_size: second dimension of matrix A.
#         bias: If true, add bias. Note that bias is not parallelized.
#         input_is_parallel: If true, we assume that the input is already split across the GPUs and we do not split again.
#         init_method: method to initialize weights. Note that bias is always set to zero.
#         stride: For the strided linear layers.
#         keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
#         skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
#         is_expert: If True, the layer is treated as an MoE expert layer
#         tp_comm_buffer_name: Communication buffer name. Not used in
#                              non-Transformer-Engine modules.
#         config: ModelParallelConfig object

#     """

#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         *,
#         config: ModelParallelConfig,
#         init_method: Callable,
#         bias: bool,
#         input_is_parallel: bool,
#         skip_bias_add: bool,
#         stride: int = 1,
#         keep_master_weight_for_test: bool = False,
#         is_expert: bool = False,
#         tp_comm_buffer_name: str = None,  # Not used
#         moe=False, enable_expert_tensor_parallelism=False,
#     ):
#         super(RowParallelLinear, self).__init__()

#         # Keep input parameters
#         self.input_size = input_size
#         self.output_size = output_size
#         self.input_is_parallel = input_is_parallel
#         self.skip_bias_add = skip_bias_add
#         self.config = config
#         self.is_expert = is_expert
#         self.expert_parallel = config.expert_model_parallel_size > 1
#         self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
#         self.sequence_parallel = config.sequence_parallel
#         if self.sequence_parallel and not self.input_is_parallel:
#             raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

#         args = get_args()
#         self.deepspeed = args.deepspeed
#         self.explicit_expert_comm = False
#         rank = get_tensor_model_parallel_rank()
#         if not args.deepspeed:
#             self.explicit_expert_comm = self.is_expert and (
#                 config.tensor_model_parallel_size > 1 or self.expert_parallel
#             )

#             # Divide the weight matrix along the last dimension.
#             if self.explicit_expert_comm and config.moe_extended_tp:
#                 world_size = get_tensor_and_expert_parallel_world_size()
#                 rank = get_tensor_and_expert_parallel_rank()
#             else:
#                 world_size = get_tensor_model_parallel_world_size()
#                 rank = get_tensor_model_parallel_rank()
#         else:
#             if moe and (not enable_expert_tensor_parallelism):
#                 world_size = 1
#             else:
#                 world_size = get_tensor_model_parallel_world_size()
#             self.is_expert_without_slicing = moe and world_size==1

#         self.input_size_per_partition = divide(input_size, world_size)

#         # Parameters.
#         # Note: torch.nn.functional.linear performs XA^T + b and as a result
#         # we allocate the transpose.
#         # Initialize weight.
#         if config.use_cpu_initialization:
#             self.weight = Parameter(
#                 torch.empty(
#                     self.output_size, self.input_size_per_partition, dtype=config.params_dtype
#                 )
#             )
#             if config.perform_initialization:
#                 self.master_weight = _initialize_affine_weight_cpu(
#                     self.weight,
#                     self.output_size,
#                     self.input_size,
#                     self.input_size_per_partition,
#                     1,
#                     init_method,
#                     stride=stride,
#                     return_master_weight=keep_master_weight_for_test,
#                     params_dtype=config.params_dtype,
#                     rank=rank,
#                     world_size=world_size,
#                 )
#         else:
#             self.weight = Parameter(
#                 torch.empty(
#                     self.output_size,
#                     self.input_size_per_partition,
#                     device=torch.cuda.current_device(),
#                     dtype=config.params_dtype,
#                 )
#             )
#             if config.perform_initialization:
#                 _initialize_affine_weight_gpu(
#                     self.weight,
#                     init_method,
#                     partition_dim=1,
#                     stride=stride,
#                     expert_parallel=(self.is_expert and self.expert_parallel),
#                 )
#         setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

#         if bias:
#             if config.use_cpu_initialization:
#                 self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
#             else:
#                 self.bias = Parameter(
#                     torch.empty(
#                         self.output_size,
#                         device=torch.cuda.current_device(),
#                         dtype=config.params_dtype,
#                     )
#                 )

#             if config.perform_initialization:
#                 # Always initialize bias to zero.
#                 with torch.no_grad():
#                     self.bias.zero_()
#             setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
#             setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
#         else:
#             self.register_parameter('bias', None)

#         self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

#         # Hook adding a default empty _extra_state for state dict
#         self._register_load_state_dict_pre_hook(
#             lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
#                 f'{prefix}_extra_state'
#             )
#         )

#     def forward(self, input_, ignore_forward=False, recompute_fwd=False):
#         """Forward of RowParallelLinear

#         Args:
#             input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

#         Returns:
#             - output
#             - bias
#         """

#         if self.config._cpu_offloading_context is not None:
#             if self.config._cpu_offloading_context.inside_context == True:
#                 assert (
#                     self.config.cpu_offloading == False
#                 ), "CPU Offloading cannot be enabled while using non-TE modules"

#         # Set up backprop all-reduce.
#         if self.input_is_parallel or (self.deepspeed and self.is_expert_without_slicing):
#             input_parallel = input_
#         else:
#             assert not self.sequence_parallel
#             input_parallel = scatter_to_tensor_model_parallel_region(input_)
#         # Matrix multiply.
#         if not self.weight.requires_grad:
#             self._forward_impl = linear_with_frozen_weight
#         else:
#             self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

#         allreduce_dgrad = False

#         output_parallel = self._forward_impl(
#             input=input_parallel,
#             weight=self.weight,
#             bias=None,
#             gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#             async_grad_allreduce=allreduce_dgrad,
#             sequence_parallel=False,
#             grad_output_buffer=None,
#             allreduce_dgrad=allreduce_dgrad,
#         )

#         # All-reduce across all the partitions.
#         if self.explicit_expert_comm:
#             assert self.skip_bias_add
#             output_ = output_parallel
#         elif self.deepspeed and self.is_expert_without_slicing: # non-expert only tensor-parallelism
#             output_ = output_parallel
#         elif self.sequence_parallel:
#             output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
#         else:
#             output_ = reduce_from_tensor_model_parallel_region(output_parallel)
#         if not self.skip_bias_add:
#             output = (output_ + self.bias) if self.bias is not None else output_
#             output_bias = None
#         else:
#             output = output_
#             output_bias = self.bias
#         return output, output_bias

#     def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
#         """Sharding along axis 1, bias not sharded"""
#         state_dict = self.state_dict(prefix='', keep_vars=True)
#         return make_sharded_tensors_for_checkpoint(
#             state_dict, prefix, {'weight': 1}, sharded_offsets
#         )

#     def set_extra_state(self, state: Any):
#         """Extra state is ignored"""

#     def get_extra_state(self) -> None:
#         """Keep compatibility with TE state dict."""
#         return None
