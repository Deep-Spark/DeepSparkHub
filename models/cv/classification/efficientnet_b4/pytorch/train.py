# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import math

import torch
import torch.utils.data

try:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
except:
    autocast = None
    scaler = None


from torch import nn
import torch.distributed as dist
import _torchvision as torchvision

import utils

from dataloader.classification import get_datasets, create_dataloader
from common_utils import LabelSmoothingCrossEntropy


def compute_loss(model, image, target, criterion):
    output = model(image)
    if not isinstance(output, (tuple, list)):
        output = [output]
    losses = []
    for out in output:
        losses.append(criterion(out, target))
    loss = sum(losses)
    return loss, output[0]


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, amp=False, use_dali=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    all_fps = []
    for data in metric_logger.log_every(data_loader, print_freq, header):
        if use_dali:
            image, target = data[0]["data"], data[0]["label"][:, 0].long()
        else:
            image, target = data
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if autocast is None or not amp:
            loss, output = compute_loss(model, image, target, criterion)
        else:
            with autocast():
                loss, output = compute_loss(model, image, target, criterion)

        optimizer.zero_grad()
        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        fps = batch_size / (end_time - start_time) * utils.get_world_size()
        metric_logger.meters['img/s'].update(fps)
        all_fps.append(fps)

    print(header, 'Avg img/s:', sum(all_fps) / len(all_fps))


def evaluate(model, criterion, data_loader, device, print_freq=100, use_dali=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, print_freq, header):
            if use_dali:
                image, target = data[0]["data"], data[0]["label"][:, 0].long()
            else:
                image, target = data
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    utils.manual_seed(args.seed, deterministic=False)
    # torch.backends.cudnn.benchmark = True

    # WARN:
    if dist.is_initialized():
        num_gpu = dist.get_world_size()
    else:
        num_gpu = 1

    global_batch_size = num_gpu * args.batch_size

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    num_classes = len(os.listdir(train_dir))
    if 0 < num_classes < 13:
        if global_batch_size > 512:
            if utils.is_main_process():
                print("WARN: Updating global batch size to 512, avoid non-convergence when training small dataset.")
            args.batch_size = 512 // num_gpu

    data_loader, data_loader_test = create_dataloader(train_dir, val_dir, args)
    
    print(f"Creating model {args.model}")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80 ,90], gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed and not args.dali:
            data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, args.amp, use_dali=args.dali)
        lr_scheduler.step()
        acc_avg = evaluate(model, criterion, data_loader_test, device=device, use_dali=args.dali)
        if acc_avg > args.acc_thresh:
            print("The accuracy has been exceeded {},and the training is terminated at epoch {}".format(args.acc_thresh, epoch))
            return

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
            epoch_total_time = time.time() - epoch_start_time
            epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
            print('epoch time {}'.format(epoch_total_time_str))

        if args.dali:
            data_loader.reset()
            data_loader_test.reset()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', 
                        # default=0.1, 
                        default=0.128, 
                        type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', 
                        # default=1e-4, 
                        default=2e-4, 
                        type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--acc-thresh',
                        default=75.0, type=float,
                        help='accuracy threshold')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')
    parser.add_argument(
        "--dali",
        help="Use dali as dataloader",
        default=False,
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int,
                        help='Local rank')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser


def get_master_addr():
    if "MASTER_ADDR" in os.environ:
        return os.environ["MASTER_ADDR"]
    return "127.0.0.1"


def check_args(args):
    master_addr = get_master_addr()
    if master_addr != "127.0.0.1" and args.dist_url != "env://":
        args.dist_url = 'tcp://' + os.environ["MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]

    return args


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args = check_args(args)
    try:
        from dltest import show_training_arguments
        show_training_arguments(args)
    except:
        pass
    main(args)
