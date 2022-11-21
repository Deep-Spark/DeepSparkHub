# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.


from torch.utils.data import DataLoader
import torch
from importlib import import_module


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        train_sampler = None
        if ((not args.cpu) and (args.num_gpu > 1)):
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
            dataloader_train = DataLoader(data_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        else:
            dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            test_sampler = None
            if ((not args.cpu) and (args.num_gpu > 1)):
                test_sampler = torch.utils.data.distributed.DistributedSampler(data_test)
                dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)
            else:
                dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader