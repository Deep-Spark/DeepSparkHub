import sys
sys.path.append("..")

import torch
from dataloaders import create_train_dataloader, create_eval_dataloader

if __name__ == "__main__":
    class Config(object):
        def __init__(self):
            pass

    config = Config()
    config.data_dir = "/home/data/perf/ssd"
    config.input_size = 300
    config.n_gpu = 1
    config.distributed = False
    config.local_rank = 0
    config.local_seed = 32
    config.num_workers = 4
    config.train_batch_size = 32
    config.eval_batch_size = 32
    config.fp16 = True
    config.fast_nms = False
    config.fast_cj = False
    config.dali = False
    config.dali_sync = False
    config.dali_cache = 0
    config.nhwc = False
    config.pad_input = False
    config.jit = False
    config.use_nvjpeg = False

    train_loader, epoch_size, train_sampler = create_train_dataloader(config)
    for batch in train_loader:
        print(len(batch))
        break
    val_loader, inv_map, cocoGt = create_eval_dataloader(config)
    from dataloaders.prefetcher import eval_prefetcher
    val_loader = eval_prefetcher(iter(val_loader),
                    torch.cuda.current_device(),
                    config.pad_input,
                    config.nhwc,
                    config.fp16)
    for batch in val_loader:
        print(len(batch))
        break
    print("finished!")