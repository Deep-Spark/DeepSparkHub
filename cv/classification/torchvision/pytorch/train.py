# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import warnings
warnings.filterwarnings('ignore')

import datetime
import os
import logging
import time

import torch
import torch.utils.data

try:
    from apex import amp as apex_amp
except:
    apex_amp = None

try:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
except:
    autocast = None
    scaler = None


from torch import nn
import torch.distributed as dist
import torchvision

import utils
from utils import (MetricLogger, SmoothedValue, accuracy, mkdir,\
                    init_distributed_mode, manual_seed,\
                    is_main_process, save_on_master, write_on_master)

from dataloader.classification import get_datasets, create_dataloader


def compute_loss(model, image, target, criterion):
    output = model(image)
    if not isinstance(output, (tuple, list)):
        output = [output]
    losses = []
    for out in output:
        losses.append(criterion(out, target))
    loss = sum(losses)
    return loss, output[0]


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, use_amp=False, use_dali=False):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    all_fps = []
    for data in metric_logger.log_every(data_loader, print_freq, header):
        if use_dali:
            image, target = data[0]["data"], data[0]["label"][:, 0].long()
        else:
            image, target = data

        start_time = time.time()
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        loss, output = compute_loss(model, image, target, criterion)

        if use_amp:
            with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        fps = batch_size / (end_time - start_time) * utils.get_world_size()
        metric_logger.meters['img/s'].update(fps)
        all_fps.append(fps)

    fps = round(sum(all_fps) / len(all_fps), 2)
    print(header, 'Avg img/s:', fps)
    return fps


def evaluate(model, criterion, data_loader, device, print_freq=100, use_dali=False):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
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

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
    return round(metric_logger.acc1.global_avg, 2)


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def create_optimzier(params, args):
    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    elif opt_name == "fused_sgd":
        from apex.optimizers import FusedSGD
        optimizer = FusedSGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    return optimizer


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    manual_seed(args.seed, deterministic=args.deterministic)

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
            if is_main_process():
                print("WARN: Updating global batch size to 512, avoid non-convergence when training small dataset.")
            args.batch_size = 512 // num_gpu

    if args.pretrained:
        num_classes = 1000
    args.num_classes = num_classes

    print("Creating model")
    if hasattr(args, "model_cls"):
        model = args.model_cls(args)
    else:
        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=num_classes)
        if args.padding_channel:
            print("WARN: Cannot convert first conv to N4HW.")

    data_loader, data_loader_test = create_dataloader(train_dir, val_dir, args)

    if args.padding_channel and isinstance(data_loader, torch.utils.data.DataLoader):
        data_loader.collate_fn = utils.nhwc_collect_fn(data_loader.collate_fn, fp16=args.amp, padding=args.padding_channel)
        data_loader_test.collate_fn = utils.nhwc_collect_fn(data_loader_test.collate_fn, fp16=args.amp, padding=args.padding_channel)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    if args.nhwc:
        model = model.cuda().to(memory_format=torch.channels_last)

    optimizer = create_optimzier(model.parameters(), args)
    if args.amp:
        model, optimizer = apex_amp.initialize(model, optimizer, opt_level="O2",
                                          loss_scale="dynamic",
                                          master_weights=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

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
    best_acc1 = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed and not args.dali:
            data_loader.sampler.set_epoch(epoch)
        fps = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, args.amp, use_dali=args.dali)
        lr_scheduler.step()
        acc1 = evaluate(model, criterion, data_loader_test, device=device, use_dali=args.dali)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = epoch
        if args.output_dir is not None:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'best.pth'.format(epoch)))
            save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'latest.pth'))
            epoch_total_time = time.time() - epoch_start_time
            epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
            print('epoch time {}'.format(epoch_total_time_str))

        if args.dali:
            data_loader.reset()
            data_loader_test.reset()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('* Acc@1: {} at epoch {}'.format(round(best_acc1, 2), best_epoch))
    print('Training time {}'.format(total_time_str))
    if args.output_dir:
        write_on_master({"Name":os.path.basename(args.output_dir),
                         "Model": args.model, "Dataset": os.path.basename(args.data_path), "AMP":args.amp,
                         "Acc@1":best_acc1, "FPS":fps, "Time": total_time_str}, os.path.join(args.output_dir, 'result.json'))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.128, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
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
        "--deterministic",
        help="Do not benchmark conv algo",
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
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # other
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision training')
    parser.add_argument('--nhwc', action='store_true', help='Use NHWC')
    parser.add_argument('--padding-channel', action='store_true', help='Padding the channels of image to 4')
    parser.add_argument('--dali-cpu', action='store_true')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--crop-size', default=224, type=int)
    parser.add_argument('--base-size', default=256, type=int)
    return parser



def check_agrs(args):
    if args.nhwc:
        args.amp = True

    if args.output_dir:
        prefix=args.output_dir
        names = [args.model, os.path.basename(args.data_path)]
        if args.amp:
            names.append("amp")
        if torch.cuda.device_count():
            names.append(f"dist_{utils.get_world_size()}x{torch.cuda.device_count()}")
        exp_dir = "_".join(map(str, names))
        args.output_dir = os.path.join(prefix, exp_dir)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    if args.amp:
        if apex_amp is None:
            raise RuntimeError("Not found apex in installed packages, cannot enable amp.")


def train_model(model_cls=None):
    args = get_args_parser().parse_args()
    check_agrs(args)

    if utils.is_main_process():
        setup_logging(args.output_dir)

    if hasattr(torch, "corex") and args.dali:
        args.dali_cpu = True

    if model_cls is not None:
        args.model_cls = model_cls

    main(args)

def setup_logging(prefix):
    if prefix:
        handlers=[
            logging.FileHandler(os.path.join(prefix, "train.log"), mode='w'),
            logging.StreamHandler(),
        ]
    else:
        handlers = None
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    check_agrs(args)
    if utils.is_main_process():
        setup_logging(args.output_dir)
    try:
        main(args)
    except Exception as e:
        logging.exception(e)
