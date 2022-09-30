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
    config.fast_nms = True
    config.fast_cj = True
    config.use_coco_ext = False
    config.dali = True
    config.dali_sync = False
    config.dali_cache = -1
    config.nhwc = True
    config.pad_input = True
    config.jit = True
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