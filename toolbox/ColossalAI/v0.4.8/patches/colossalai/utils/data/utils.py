import torch
import torch.distributed as dist


def get_group_world_size(group):
    """Return world size for parallel group."""
    return torch.distributed.get_world_size(group=group)


def get_group_src_rank(group):
    """Calculate the global rank corresponding to the first local rank in the parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_group_world_size(group)
    return (global_rank // local_world_size) * local_world_size


def get_group_rank(group):
    """Return my rank for the parallel group."""
    return torch.distributed.get_rank(group=group)


def is_pipeline_first_stage(group):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""

    return get_group_rank(group) == 0


def is_pipeline_last_stage(group):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""

    return get_group_rank(group) == (get_group_world_size(group) - 1)


def get_batch_on_this_tp_rank(args, data_iterator, tp_group, pp_group):
    """
        broadcast batch data from src rank to other ranks in tp group
    """
    # No need to broadcast in the middle stage of pipeline
    if not is_pipeline_first_stage(pp_group) and not is_pipeline_last_stage(pp_group):
        return None

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, get_group_src_rank(tp_group), group=tp_group)

    if dist.get_rank(tp_group) == 0:
        data = next(data_iterator)
        input_ids = data["input_ids"].cuda(non_blocking=True)
        attention_mask = data["attention_mask"].cuda(non_blocking=True)
        assert attention_mask.dtype == torch.bool
        labels = data["labels"].cuda(non_blocking=True)
    else:
        input_ids = torch.empty((args.batch_size, args.max_length), dtype=torch.int64, device=torch.cuda.current_device())
        attention_mask = torch.empty((args.batch_size, args.max_length), dtype=torch.bool, device=torch.cuda.current_device())
        labels = torch.empty((args.batch_size, args.max_length), dtype=torch.int64, device=torch.cuda.current_device())

    if args.pp == 1:
        _broadcast(input_ids)
        _broadcast(attention_mask)
        _broadcast(labels)
    elif is_pipeline_first_stage(pp_group):
        if dist.get_rank(tp_group) != 0:
            labels=None
        _broadcast(input_ids)
        _broadcast(attention_mask)
    elif is_pipeline_last_stage(pp_group):
        if dist.get_rank(tp_group) != 0:
            input_ids=None
        _broadcast(attention_mask)
        _broadcast(labels)

    batch = {"input_ids": input_ids, "attention_mask":attention_mask, "labels":labels}

    return batch