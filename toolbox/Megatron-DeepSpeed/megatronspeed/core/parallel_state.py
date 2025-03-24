"""Model and data parallel groups."""

import os
from functools import wraps
import warnings
from datetime import timedelta
from typing import List, Optional

import torch

from megatron.training.global_vars import get_args

import megatron.core.parallel_state as ps
# from megatron.core.parallel_state import (
#     _TENSOR_MODEL_PARALLEL_GROUP,
#     _PIPELINE_MODEL_PARALLEL_GROUP,
#     _MODEL_PARALLEL_GROUP,
#     _MODEL_AND_EXPERT_PARALLEL_GROUP,
#     _EMBEDDING_GROUP,
#     _EMBEDDING_AR_GROUP,
#     _POSITION_EMBEDDING_GROUP,
#     _DATA_PARALLEL_GROUP,
#     _DATA_PARALLEL_GROUP_GLOO,
#     _TENSOR_AND_DATA_PARALLEL_GROUP,
#     _EXPERT_MODEL_PARALLEL_GROUP,
#     _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK,
#     _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE,
#     _PIPELINE_MODEL_PARALLEL_SPLIT_RANK,
#     _EMBEDDING_GLOBAL_RANKS,
#     _POSITION_EMBEDDING_GLOBAL_RANKS,
#     _PIPELINE_GLOBAL_RANKS,
#     _TENSOR_GLOBAL_RANKS,
#     _DATA_PARALLEL_GLOBAL_RANKS,
#     _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS,
#     _DATA_PARALLEL_DEVICE_GROUP,
# )
from megatron.core.parallel_state import (
    get_nccl_options,
    RankGenerator,
    get_data_parallel_group,
    get_context_parallel_world_size,
    _set_global_memory_buffer,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)

from megatronspeed.megatron_adaptor import get_megatronspeed_args

# For DeepSpeed's sequence parallel
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# This group includes processes for both data and sequence parallelisms.
# We use this group to reduce gradients and shard parameters and optimizer stages for ZeRO.
_SEQUENCE_DATA_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
_SEQUENCE_DATA_PARALLEL_RANK = None

def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        sequence_parallel_size: int = 1,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
    ):
        print("megatrospeed initialize_model_parallel_wrapper.")
        try:
            args = get_args()
        except AssertionError:
            args = get_megatronspeed_args()

        # import pdb
        # if torch.distributed.get_rank() == 0:
        #     pdb.set_trace()

        # if not args.deepspeed:
        #     initialize_model_parallel(
        #         tensor_model_parallel_size,
        #         pipeline_model_parallel_size,
        #         virtual_pipeline_model_parallel_size,
        #         pipeline_model_parallel_split_rank,
        #         use_sharp,
        #         context_parallel_size,
        #         expert_model_parallel_size,
        #         nccl_communicator_config_path,
        #         distributed_timeout_minutes,
        #         order,
        #     )
        #     return
        
        print(f"tensor_model_parallel_size: {tensor_model_parallel_size}, pipeline_model_parallel_size: {pipeline_model_parallel_size}")

        if sequence_parallel_size > 1:
            assert args.context_parallel_size <= 1, "Megatron-lm CP is not compatible with Deppspeed SP"

        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()

        if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
            )

        enable_ds_sequence_parallel = sequence_parallel_size > 1
        if enable_ds_sequence_parallel:
            assert tensor_model_parallel_size == 1 and pipeline_model_parallel_size == 1, \
            'DeepSpeed\'s sequence parallel does not work with tensor parallel or pipeline parallel'

            if world_size % sequence_parallel_size != 0:
                raise RuntimeError(
                    f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})"
                )

        data_parallel_size: int = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * sequence_parallel_size)
        sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        num_data_parallel_groups: int = world_size // data_parallel_size
        num_sequence_parallel_groups: int = world_size // sequence_parallel_size
        num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 2:
                raise RuntimeError("pipeline-model-parallel size should be greater than 2 with " "interleaved schedule")
            ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        if pipeline_model_parallel_split_rank is not None:
            ps._PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                )

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        rank_generator = RankGenerator(
            tp=tensor_model_parallel_size,
            ep=expert_model_parallel_size,
            dp=data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
        )
        timeout = timedelta(minutes=distributed_timeout_minutes)

        # Build the data-parallel groups.
        assert ps._DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
        all_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups

            if sequence_parallel_size > 1:
                tp_or_sp_size = sequence_parallel_size
            else:
                tp_or_sp_size = tensor_model_parallel_size

            for j in range(tp_or_sp_size):
                ranks = range(start_rank + j, end_rank, tp_or_sp_size)
                all_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(ranks)
                if getattr(args, "use_distributed_optimizer", None):
                    group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                else:
                    group_gloo = None
                if rank in ranks:
                    ps._DATA_PARALLEL_GROUP = group
                    ps._DATA_PARALLEL_GROUP_GLOO = group_gloo
                    ps._DATA_PARALLEL_GLOBAL_RANKS = ranks
        for ranks_with_cp in rank_generator.get_ranks('dp-cp'):
            group_with_cp = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
            )
            group_with_cp_gloo = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, backend="gloo"
            )
            if rank in ranks_with_cp:
                ps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        # Build the sequence parallel groups.
        global _SEQUENCE_PARALLEL_GROUP
        assert _SEQUENCE_PARALLEL_GROUP is None, \
            'sequence parallel group is already initialized'
        for i in range(num_sequence_parallel_groups):
            ranks = range(i * sequence_parallel_size,
                        (i + 1) * sequence_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _SEQUENCE_PARALLEL_GROUP = group

        # Build the sequence data parallel groups.
        global _SEQUENCE_DATA_PARALLEL_GROUP
        assert _SEQUENCE_DATA_PARALLEL_GROUP is None, \
            'sequence data parallel group is already initialized'
        all_data_sequence_parallel_group_ranks = []
        if enable_ds_sequence_parallel:
            for i in range(num_sequence_data_parallel_groups):
                ranks = range(i * sequence_data_parallel_size,
                            (i + 1) * sequence_data_parallel_size)
                group = torch.distributed.new_group(ranks)
                all_data_sequence_parallel_group_ranks.append(list(ranks))
                if rank in ranks:
                    _SEQUENCE_DATA_PARALLEL_GROUP = group
        else:
            _SEQUENCE_DATA_PARALLEL_GROUP = ps._DATA_PARALLEL_GROUP

        # Build the context-parallel groups.
        assert ps._CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
        for ranks in rank_generator.get_ranks('cp'):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._CONTEXT_PARALLEL_GROUP = group
                ps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

        # Build the model-parallel groups.
        assert ps._MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
        num_model_parallel_groups = sequence_data_parallel_size if enable_ds_sequence_parallel else data_parallel_size
        model_parallel_group_ranks = all_data_sequence_parallel_group_ranks if enable_ds_sequence_parallel else all_data_parallel_group_ranks
        for i in range(num_model_parallel_groups):
            ranks = [parallel_group_ranks[i] for parallel_group_ranks in model_parallel_group_ranks]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                ps._MODEL_PARALLEL_GROUP = group

        # Build the model-parallel groups with expert parallel
        assert (
            ps._MODEL_AND_EXPERT_PARALLEL_GROUP is None
        ), 'model and expert parallel group is already initialized'
        for ranks in rank_generator.get_ranks('tp-ep-pp', independent_ep=True):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('mp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._MODEL_AND_EXPERT_PARALLEL_GROUP = group

        # Build the tensor model-parallel groups.
        assert ps._TENSOR_MODEL_PARALLEL_GROUP is None, 'tensor model parallel group is already initialized'
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                ps._TENSOR_MODEL_PARALLEL_GROUP = group
                ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks


        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, 'pipeline model parallel group is already initialized'
        assert ps._EMBEDDING_GROUP is None, 'embedding group is already initialized'
        assert ps._POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device(),])
                ps._PIPELINE_MODEL_PARALLEL_GROUP = group
                ps._PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank], ranks[-1]]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                ps._EMBEDDING_GROUP = group
            if rank in ranks:
                ps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(position_embedding_ranks)
            if rank in position_embedding_ranks:
                ps._POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                ps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        # Build the tensor + data parallel groups.
        assert (
            ps._TENSOR_AND_DATA_PARALLEL_GROUP is None
        ), 'Tensor + data parallel group is already initialized'
        for ranks in rank_generator.get_ranks('tp-dp-cp'):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
        for ranks in rank_generator.get_ranks('tp-dp'):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._TENSOR_AND_DATA_PARALLEL_GROUP = group

        assert (
            ps._TENSOR_AND_CONTEXT_PARALLEL_GROUP is None
        ), 'Tensor + context parallel group is already initialized'
        for ranks in rank_generator.get_ranks('tp-cp'):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

        # Build the tensor + expert parallel groups
        assert ps._EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
        assert (
            ps._TENSOR_AND_EXPERT_PARALLEL_GROUP is None
        ), 'Tensor + expert parallel group is already initialized'
        assert (
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP is None
        ), 'Data modulo expert group is already initialized'
        assert (
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP is None
        ), 'Data modulo expert group with context parallel is already initialized'

        for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        for ranks in rank_generator.get_ranks('ep', independent_ep=True):
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                ps._EXPERT_MODEL_PARALLEL_GROUP = group

        for ranks in rank_generator.get_ranks('dp', independent_ep=True):
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                ps._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

        for ranks in rank_generator.get_ranks('dp-cp', independent_ep=True):
            # Lazy initialization of the group
            if ps._CONTEXT_PARALLEL_GROUP and get_context_parallel_world_size() > 1:
                group = torch.distributed.new_group(
                    ranks,
                    timeout=timeout,
                    pg_options=get_nccl_options('dp_modulo_exp_cp', nccl_comm_cfgs),
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            else:
                group = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP
                group_gloo = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
            if rank in ranks:
                ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP = group
                ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo

        # Initialize global memory buffer
        # This isn't really "parallel state" but there isn't another good place to
        # put this. If we end up with a more generic initialization of megatron-core
        # we could stick it there
        _set_global_memory_buffer()

        print(f"_PIPELINE_MODEL_PARALLEL_GROUP: {ps._PIPELINE_MODEL_PARALLEL_GROUP}")

    return wrapper

def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or \
        ps._DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_data_parallel_is_initialized():
    """Check if sequence data parallel groups are initialized."""
    if _SEQUENCE_DATA_PARALLEL_GROUP is None:
        return False
    return True

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, \
        'sequence data parallel group is not initialized'
    return _SEQUENCE_DATA_PARALLEL_GROUP

def set_sequence_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = world_size

def set_sequence_data_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    _SEQUENCE_DATA_PARALLEL_WORLD_SIZE = world_size

def get_model_parallel_world_size():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_world_size is only supported if PP is disabled"
    return get_tensor_model_parallel_world_size()

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def get_sequence_data_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    if _SEQUENCE_DATA_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_DATA_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())

def get_model_parallel_rank():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_rank is only supported if PP is disabled"
    return get_tensor_model_parallel_rank()


def set_sequence_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_PARALLEL_RANK
    _SEQUENCE_PARALLEL_RANK = rank


def set_sequence_data_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    _SEQUENCE_DATA_PARALLEL_RANK = rank


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_data_parallel_rank():
    """Return my rank for the sequence data parallel group."""
    global _SEQUENCE_DATA_PARALLEL_RANK
    if _SEQUENCE_DATA_PARALLEL_RANK is not None:
        return _SEQUENCE_DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_sequence_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def destroy_model_parallel_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()
        global _SEQUENCE_PARALLEL_GROUP
        _SEQUENCE_PARALLEL_GROUP = None
        global _SEQUENCE_DATA_PARALLEL_GROUP
        _SEQUENCE_DATA_PARALLEL_GROUP = None

    return wrapper
