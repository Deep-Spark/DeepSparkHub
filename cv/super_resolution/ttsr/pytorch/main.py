from option import args
from utils import mkExpDir,Logger
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if ((not args.cpu) and (args.num_gpu > 1)) and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        env_rank = int(os.environ["RANK"])
        env_world_size = int(os.environ["WORLD_SIZE"])
        env_gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    dist_backend = "nccl"
    print('| distributed init (rank {}) (size {})'.format(env_rank,env_world_size), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method='env://',
                                         world_size=env_world_size, rank=env_rank)

    torch.cuda.set_device(env_gpu)
    torch.distributed.barrier()
    setup_for_distributed(env_rank == 0)


if __name__ == '__main__':
    init_distributed_mode(args)
    ### make save_dir
    if ((not args.cpu) and (args.num_gpu > 1)):
        if int(os.environ["RANK"]) == 0:
            mkExpDir(args)
    else:
        mkExpDir(args)

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()
    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args).to(device)
    model_without_ddp = _model
    if ((not args.cpu) and (args.num_gpu > 1)):
        cur_gpu = int(os.environ['LOCAL_RANK'])
        _model = nn.parallel.DistributedDataParallel(_model, device_ids=[cur_gpu])
        model_without_ddp = _model.module

    ### loss
    _loss_all = get_loss_dict(args)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
