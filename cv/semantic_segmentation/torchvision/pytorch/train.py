# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
from torch import nn
import torch.nn.functional as TF

try:
    from apex import amp as apex_amp
except:
    apex_amp = None

import utils
from dataloader.segmentation import get_dataset

try:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
except:
    autocast = None
    scaler = None


def criterion(inputs, target):
    if isinstance(inputs, (tuple, list)):
        inputs = {str(i): x for i, x in enumerate(inputs)}
        inputs["out"] = inputs.pop("0")

    if not isinstance(inputs, dict):
        inputs = dict(out=inputs)

    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    loss = losses.pop("out")
    return loss + 0.5 * sum(losses.values())


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, dict):
                output = output['out']
            if isinstance(output, (tuple, list)):
                output = output[0]

            if output.shape[2:] != image.shape[2:]:
                output = TF.upsample(output, image.shape[2:], mode="bilinear")

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer,
                    data_loader, lr_scheduler,
                    device, epoch, print_freq,
                    use_amp=False, use_nhwc=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    all_fps = []
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        output = model(image)
        loss = criterion(output, target)

        if use_amp:
            with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()
        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        fps = image.shape[0] / (end_time - start_time) * utils.get_world_size()
        metric_logger.meters['img/s'].update(fps)
        all_fps.append(fps)

    print(header, 'Avg img/s:', sum(all_fps) / len(all_fps))


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train",
                                       crop_size=args.crop_size, base_size=args.base_size)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val",
                                  crop_size=args.crop_size, base_size=args.base_size)
    args.num_classes = num_classes

    if args.nhwc:
        collate_fn = utils.nhwc_collate_fn(fp16=args.amp, padding_channel=args.padding_channel)
    else:
        collate_fn = utils.collate_fn

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=collate_fn)

    if hasattr(args, "model_cls"):
        model = args.model_cls(args)
    else:
        model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                     aux_loss=args.aux_loss,
                                                                     pretrained=args.pretrained)
        if args.padding_channel:
            if hasattr(model, "backbone") and hasattr(model.backbone, "conv1"):
                model.backbone.conv1 = utils.padding_conv_channel_to_4(model.backbone.conv1)
            else:
                print("WARN: Cannot convert first conv to N4HW.")
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.nhwc:
        model = model.cuda().to(memory_format=torch.channels_last)

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = apex_amp.initialize(model, optimizer, opt_level="O2",
                                          loss_scale=args.loss_scale,
                                          master_weights=True)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq,
                        args.amp, args.nhwc)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        if args.output_dir is not None:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print('epoch time {}'.format(epoch_total_time_str))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset path')
    parser.add_argument('--dataset', default='camvid', help='dataset name')
    parser.add_argument('--model', default='deeplabv3_resnet50', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
    # distributed training parameters
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision training')
    parser.add_argument('--padding-channel', action='store_true', help='Padding the channels of image to 4')
    parser.add_argument('--loss_scale', default="dynamic", type=str)
    parser.add_argument('--nhwc', action='store_true', help='Use NHWC')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--crop-size', default=512, type=int)
    parser.add_argument('--base-size', default=540, type=int)
    return parser


def check_agrs(args):
    try:
        args.loss_scale = float(args.loss_scale)
    except: pass

    if args.padding_channel:
        if not args.nhwc:
            print("Turning nhwc when padding the channel of image.")
            args.nhwc = True

    if args.amp:
        if apex_amp is None:
            raise RuntimeError("Not found apex in installed packages, cannot enable amp.")


def train_model(model_cls=None):
    args = get_args_parser().parse_args()
    check_agrs(args)
    if model_cls is not None:
        args.model_cls = model_cls
    main(args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    check_agrs(args)
    main(args)
