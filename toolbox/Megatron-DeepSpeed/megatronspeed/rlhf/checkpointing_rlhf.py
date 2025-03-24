# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import sys
import torch
import random
import numpy as np

from megatron.core import mpu, tensor_parallel
from megatron.training.global_vars import get_args, set_args, set_rlhf_args
from megatron.training.utils import unwrap_model, print_rank_0



def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok = True)


def get_distributed_optimizer_checkpoint_name(model_checkpoint_name):
    return os.path.join(os.path.dirname(model_checkpoint_name),
                        "distrib_optim.pt")


def get_checkpoint_name(checkpoints_path, pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None):
    """Determine the directory name for this rank's checkpoint."""

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path,
                f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    return os.path.join(common_path, "model_optim_rng.pt")


def get_checkpoint_tracker_filename(checkpoints_path):

    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def get_rng_state():
    """ collect rng state across data parallel ranks """
    args = get_args()
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    rng_state_list = None
    if torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1 and \
            args.data_parallel_random_init:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    return rng_state_list


def set_args_from_state_dict(args, state_dict, rlhf_training=False):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """

    checkpoint_args = state_dict['args']
    args.iteration = state_dict['iteration']

    assert getattr(checkpoint_args, "tensor_model_parallel_size", None) == getattr(args, "tensor_model_parallel_size", None)
    assert getattr(checkpoint_args, "pipeline_model_parallel_size", None) == getattr(args, "pipeline_model_parallel_size", None)
    assert getattr(checkpoint_args, "virtual_pipeline_model_parallel_size", None) == getattr(args, "virtual_pipeline_model_parallel_size", None)
    assert getattr(checkpoint_args, "num_layers_per_virtual_pipeline_stage", None) == getattr(args, "num_layers_per_virtual_pipeline_stage", None)

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, 'disable_bias_linear'):
        setattr(checkpoint_args, 'add_bias_linear', not getattr(checkpoint_args, 'disable_bias_linear'))

    def _set_arg(arg_name, force=False):
        if not force and getattr(args, arg_name, None) is not None:
            return

        checkpoint_value = getattr(checkpoint_args, arg_name, None)
        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg('num_layers', force=True)
    _set_arg('hidden_size', force=True)
    _set_arg('ffn_hidden_size', force=True)
    # _set_arg('seq_length', force=True)
    _set_arg('num_attention_heads', force=True)
    _set_arg('num_query_groups', force=True)
    _set_arg('group_query_attention', force=True)
    _set_arg('kv_channels', force=True)
    _set_arg('max_position_embeddings', force=True)
    _set_arg('position_embedding_type', force=True)
    _set_arg('add_position_embedding', force=True)
    _set_arg('use_rotary_position_embeddings', force=True)
    _set_arg('rotary_percent', force=True)
    _set_arg('add_bias_linear', force=True)
    _set_arg('swiglu', force=True)
    _set_arg('untie_embeddings_and_output_weights', force=True)
    _set_arg('apply_layernorm_1p', force=True)
    _set_arg('normalization', force=True)
    _set_arg('tokenizer_type', force=True)
    _set_arg('padded_vocab_size', force=True)

    # set globla args to current args
    if rlhf_training:
        set_rlhf_args(args)
    else:
        set_args(args)


def load_state_dict(ckpt_dir):
    """ Load the base state_dict from the given directory
    """
    checkpoint_file = get_checkpoint_name(ckpt_dir)

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_file, map_location='cpu')
    except BaseException as e:
        print_rank_0(f'Could not load the checkpoint, {e}, exiting')
        sys.exit()

    return state_dict


def load_state_dict_into_model(model, state_dict, strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, model_prefix):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    save_path = os.path.join(args.save, model_prefix)
    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, save_path))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state()

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(save_path)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        if len(model) == 1:
            state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = \
                    model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        # Save.
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                 .format(iteration, save_path))

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
       or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(save_path)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

