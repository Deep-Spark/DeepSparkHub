import contextlib
from importlib.metadata import version

import torch
from pkg_resources import packaging
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from megatron.training.global_vars import get_args
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import safely_set_viewless_tensor_data
import megatron.core.tensor_parallel.random as tp_random
from megatron.core.tensor_parallel.random import (
    _set_cuda_rng_state,
    initialize_rng_tracker,
    CheckpointFunction
)
from megatron.core.tensor_parallel.utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

from megatronspeed.training.memory import allocate_mem_buff

import deepspeed
from deepspeed.accelerator import get_accelerator

# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None


def init_checkpointed_activations_memory_buffer():
    """Initializ the memory buffer for the checkpointed activations."""
    args = get_args()

    per_layer = args.micro_batch_size * args.max_position_embeddings * \
                args.hidden_size // args.tensor_model_parallel_size
    assert args.num_layers % args.checkpoint_num_layers == 0, \
        'number of layers is not divisible by checkpoint-num-layers'
    num_checkpointer_layers = args.num_layers // args.checkpoint_num_layers
    numel = per_layer * num_checkpointer_layers
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float

    global _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER
    assert _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is None, \
        'checkpointed activations memory buffer is already allocated.'
    _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = allocate_mem_buff(
        'checkpointed activations', numel, dtype, track_usage=False)


def reset_checkpointed_activations_memory_buffer():
    """Reset the memory used for checkpointing."""
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.reset()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    if deepspeed.checkpointing.is_configured():
        return deepspeed.checkpointing.get_cuda_rng_tracker()
    
    initialize_rng_tracker()
    return tp_random._CUDA_RNG_STATE_TRACKER

def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    if deepspeed.checkpointing.is_configured():
        return deepspeed.checkpointing.model_parallel_cuda_manual_seed(seed)
    
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(), get_tensor_model_parallel_rank(),
                  get_data_parallel_rank(), tensor_model_parallel_seed,
                  data_parallel_seed), flush=True)

    initialize_rng_tracker()
    tp_random._CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    tp_random._CUDA_RNG_STATE_TRACKER.add(tp_random._DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # and model parallel state.
    tp_random._CUDA_RNG_STATE_TRACKER.add(tp_random._MODEL_PARALLEL_RNG_TRACKER_NAME,
                                          tensor_model_parallel_seed)

    expert_parallel_seed = (
        seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()
    )
    tp_random._CUDA_RNG_STATE_TRACKER.add(tp_random._EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)


def model_parallel_reconfigure_tp_seed(seed):
    if deepspeed.checkpointing.is_configured():
        return deepspeed.checkpointing.model_parallel_reconfigure_tp_seed(seed)

    model_parallel_seed = seed + 2718 + get_tensor_model_parallel_rank()
    with tp_random._CUDA_RNG_STATE_TRACKER.fork():
        get_accelerator().manual_seed(model_parallel_seed)

def checkpoint_function_forward(ctx, run_function, distribute_saved_activations, *args):
    ctx.run_function = run_function
    ctx.distribute_saved_activations \
        = distribute_saved_activations

    # Copy the rng states.
    ctx.fwd_cpu_rng_state = torch.get_rng_state()
    ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
    ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    with torch.no_grad():
        outputs = run_function(*args, is_recompute_forward=False)

    # Divide hidden states across model parallel group and only keep
    # the chunk corresponding to the current rank.
    if distribute_saved_activations:
        ctx.input_0_shape = args[0].data.shape
        safely_set_viewless_tensor_data(
            args[0],
            split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

    # HACK: currently when DeepSpeed is used, we always set
    # distribute_saved_activations to false, and use the following older
    # activation checkpointing mechanisms
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        ctx.input_0_shape = args[0].data.shape
        args[0].data = split_tensor_into_1d_equal_chunks(args[0].data)
        args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(
            args[0].data)

    # Store everything.
    ctx.save_for_backward(*args)

    return outputs

def checkpoint_function_backward(ctx, *args):
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError("Checkpointing is not compatible with .grad(), "
                            "please use .backward() if possible")
    inputs = ctx.saved_tensors
    if ctx.distribute_saved_activations:
        safely_set_viewless_tensor_data(
            inputs[0],
            gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))
    # HACK: currently when DeepSpeed is used, we always set
    # distribute_saved_activations to false, and use the following older
    # activation checkpointing mechanisms
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        inputs[0].data = gather_split_1d_tensor(inputs[0].data)
        inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

    # Store the current states.
    bwd_cpu_rng_state = torch.get_rng_state()
    bwd_cuda_rng_state = torch.cuda.get_rng_state()
    bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    # Set the states to what it used to be before the forward pass.
    torch.set_rng_state(ctx.fwd_cpu_rng_state)
    _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

    # Compute the forward pass.
    detached_inputs = detach_variable(inputs)
    with torch.enable_grad():
        outputs = ctx.run_function(*detached_inputs, is_recompute_forward=True)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    elif len(outputs) == 2 and isinstance(outputs[1], torch.Tensor) and \
            torch.equal(outputs[1], torch.tensor(0).to(torch.cuda.current_device())):
        # a hacky solution to overcome issue when running old script examples/pretrain_gpt_distributed.sh
        outputs = (outputs[0],)
    # filter out non tensor outputs for backward pass
    outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
    torch.autograd.backward(outputs, args)
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                    for inp in detached_inputs)
    return (None, None) + grads

# class CheckpointFunction(torch.autograd.Function):
#     """Checkpoint Function 

#     This function is adapted from torch.utils.checkpoint with two main changes:
#     1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
#     2) the states in the model parallel tracker are also properly tracked/set/reset.
#     """

#     @staticmethod
#     def forward(ctx, run_function, distribute_saved_activations, *args):
#         ctx.run_function = run_function
#         ctx.distribute_saved_activations \
#             = distribute_saved_activations

#         # Copy the rng states.
#         ctx.fwd_cpu_rng_state = torch.get_rng_state()
#         ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
#         ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

#         with torch.no_grad():
#             outputs = run_function(*args, is_recompute_forward=False)

#         # Divide hidden states across model parallel group and only keep
#         # the chunk corresponding to the current rank.
#         if distribute_saved_activations:
#             ctx.input_0_shape = args[0].data.shape
#             safely_set_viewless_tensor_data(
#                 args[0],
#                 split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

#         # HACK: currently when DeepSpeed is used, we always set
#         # distribute_saved_activations to false, and use the following older
#         # activation checkpointing mechanisms
#         if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
#             ctx.input_0_shape = args[0].data.shape
#             args[0].data = split_tensor_into_1d_equal_chunks(args[0].data)
#             args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(
#                 args[0].data)

#         # Store everything.
#         ctx.save_for_backward(*args)

#         return outputs

#     @staticmethod
#     def backward(ctx, *args):
#         if not torch.autograd._is_checkpoint_valid():
#             raise RuntimeError("Checkpointing is not compatible with .grad(), "
#                                "please use .backward() if possible")
#         inputs = ctx.saved_tensors
#         if ctx.distribute_saved_activations:
#             safely_set_viewless_tensor_data(
#                 inputs[0],
#                 gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))
#         # HACK: currently when DeepSpeed is used, we always set
#         # distribute_saved_activations to false, and use the following older
#         # activation checkpointing mechanisms
#         if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
#             inputs[0].data = gather_split_1d_tensor(inputs[0].data)
#             inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

#         # Store the current states.
#         bwd_cpu_rng_state = torch.get_rng_state()
#         bwd_cuda_rng_state = torch.cuda.get_rng_state()
#         bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

#         # Set the states to what it used to be before the forward pass.
#         torch.set_rng_state(ctx.fwd_cpu_rng_state)
#         _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
#         get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

#         # Compute the forward pass.
#         detached_inputs = detach_variable(inputs)
#         with torch.enable_grad():
#             outputs = ctx.run_function(*detached_inputs, is_recompute_forward=True)

#         # Set the states back to what it was at the start of this function.
#         torch.set_rng_state(bwd_cpu_rng_state)
#         _set_cuda_rng_state(bwd_cuda_rng_state)
#         get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

#         if isinstance(outputs, torch.Tensor):
#             outputs = (outputs,)
#         elif len(outputs) == 2 and isinstance(outputs[1], torch.Tensor) and \
#                 torch.equal(outputs[1], torch.tensor(0).to(torch.cuda.current_device())):
#             # a hacky solution to overcome issue when running old script examples/pretrain_gpt_distributed.sh
#             outputs = (outputs[0],)
#         # filter out non tensor outputs for backward pass
#         outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
#         torch.autograd.backward(outputs, args)
#         grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
#                       for inp in detached_inputs)
#         return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    if deepspeed.checkpointing.is_configured():
        return deepspeed.checkpointing.checkpoint(function, *args)
    
    return CheckpointFunction.apply(function,
                                    distribute_saved_activations, *args)
