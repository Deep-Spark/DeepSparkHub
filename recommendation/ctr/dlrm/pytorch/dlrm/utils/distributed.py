import os
import torch
import torch.distributed as dist


def setup_distributed_print(enable):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if enable or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(backend="nccl", use_gpu=True):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Not using distributed mode')
        return None, 1, None

    if use_gpu:
        torch.cuda.set_device(gpu)

    print('| distributed init (rank {})'.format(rank), flush=True)
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method='env://')

    return rank, world_size, gpu
