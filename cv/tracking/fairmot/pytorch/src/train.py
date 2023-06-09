# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) ifzhang. All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from track import run_track_val

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    num_gpus =len(opt.gpus_str.split(','))
    if num_gpus == 1:
        opt.is_ddp = -1
    else:
        opt.is_ddp = 1
        opt.ddp_gpus = num_gpus

    if (opt.is_ddp==1) and (opt.local_rank==0): # DDP
        logger = Logger(opt)
    elif (opt.is_ddp==1):
        logger = None
    else:
        logger = Logger(opt)

    if (opt.is_dp==1): # DP
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    if opt.is_ddp==1:  # DDP
        local_rank = opt.local_rank
        # DDP：DDP backend初始化
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend = opt.ddp_backend)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size= int(opt.batch_size/opt.ddp_gpus),
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        if (opt.is_ddp==1): # DDP
            train_loader.sampler.set_epoch(epoch)
        log_dict_train, _ = trainer.train(epoch, train_loader)
        if (opt.is_ddp==1) and (opt.local_rank==0):
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
        elif (opt.is_ddp==1):
            pass
        else: # DP
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
        if (opt.is_ddp==1) and (opt.local_rank==0): # DDP
            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                        epoch, model, optimizer)
            else:
                save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                        epoch, model, optimizer)
        elif (opt.is_ddp==1):
            pass
        else:
            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                        epoch, model, optimizer)
            else:
                save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                        epoch, model, optimizer)
        if (opt.is_ddp==1) and (opt.local_rank==0): # DDP
            logger.write('\n')
        elif (opt.is_ddp==1):
            pass
        else:            
            logger.write('\n')
        if (opt.is_ddp==1) and (opt.local_rank==0): # DDP
            if epoch in opt.lr_step:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if epoch % 5 == 0 or epoch >= 25:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)
        elif (opt.is_ddp==1):
            pass
        else:
            if epoch in opt.lr_step:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if epoch % 5 == 0 or epoch >= 25:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                        epoch, model, optimizer)

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            if (opt.is_ddp!=1) or ((opt.is_ddp==1) and (opt.local_rank==0)):
                val_opt = opts().init()
                val_model_path = os.path.join(opt.save_dir, 'model_val.pth')
                save_model(val_model_path, epoch, model, optimizer)
                val_opt.val_mot17 = True
                val_opt.load_model = val_model_path
                val_opt.conf_thres = 0.4
                val_opt.data_dir = './data/MOT'
                run_track_val(val_opt)

    if (opt.is_ddp==1) and (opt.local_rank==0): # DDP
        logger.close()
    elif (opt.is_ddp==1):
        pass
    else:
        logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
