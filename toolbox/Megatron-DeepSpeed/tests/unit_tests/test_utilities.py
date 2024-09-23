import os
import torch
import megatron_ds.core.parallel_state as ps

class Utils:
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    def __init__():
        pass

    @staticmethod
    def initialize_distributed():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'Initializing torch.distributed with rank: {rank}, world_size: {world_size}')
        torch.cuda.set_device(rank % torch.cuda.device_count())
        # init_method = 'tcp://'
        # master_ip = os.getenv('MASTER_ADDR', 'localhost')
        # master_port = os.getenv('MASTER_PORT', '6000')
        # init_method += master_ip + ':' + master_port
        # torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=init_method)
        torch.distributed.init_process_group(backend='nccl')
        # local_rank = torch.distributed.get_rank()
        # torch.cuda.set_device(local_rank)
        
    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        # torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1, virtual_pipeline_model_parallel_size = None, pipeline_model_parallel_split_rank = None):
        ps.destroy_model_parallel()
        if not torch.distributed.is_initialized():
            Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank)