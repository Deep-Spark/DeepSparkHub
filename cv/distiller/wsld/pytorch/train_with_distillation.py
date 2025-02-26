# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

import gc
import os
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn

import models.MobileNet as Mov
import models.ResNet as ResNet
from imagenet_train_cfg import cfg as config
from dataset import imagenet_data
from dataset.prefetch_data import data_prefetcher
from tools import utils

warnings.filterwarnings("ignore")

best_err1 = 100
best_err5 = 100

def main():
    global best_err1, best_err5

    if config.train_params.use_seed:
        utils.set_seed(config.train_params.seed)

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(config.data.data_path, 'train'),
                                        testFolder=os.path.join(config.data.data_path, 'val'),
                                        num_workers=config.data.num_workers,
                                        type_of_data_augmentation=config.data.type_of_data_aug,
                                        data_config=config.data)

    train_loader, val_loader = imagenet.getTrainTestLoader(config.data.batch_size)

    if config.net_type == 'mobilenet':
        t_net = ResNet.resnet50(pretrained=True)
        s_net = Mov.MobileNet()
    elif config.net_type == 'resnet':
        t_net = ResNet.resnet34(pretrained=True)
        s_net = ResNet.resnet18(pretrained=False)
    else:
        print('undefined network type !!!')
        raise RuntimeError('%s does not support' % config.net_type)

    import knowledge_distiller
    d_net = knowledge_distiller.WSLDistiller(t_net, s_net)

    print('Teacher Net: ')
    print(t_net)
    print('Student Net: ')
    print(s_net)
    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

    t_net = torch.nn.DataParallel(t_net)
    s_net = torch.nn.DataParallel(s_net)
    d_net = torch.nn.DataParallel(d_net)

    if config.optim.if_resume:
        checkpoint = torch.load(config.optim.resume_path)
        d_net.module.load_state_dict(checkpoint['train_state_dict'])
        best_err1 = checkpoint['best_err1']
        best_err5 = checkpoint['best_err5']
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0


    t_net = t_net.cuda()
    s_net = s_net.cuda()
    d_net = d_net.cuda()

    ### choose optimizer parameters

    optimizer = torch.optim.SGD(list(s_net.parameters()), config.optim.init_lr,
                                momentum=config.optim.momentum, weight_decay=config.optim.weight_decay, nesterov=True)

    cudnn.benchmark = True
    cudnn.enabled = True

    print('Teacher network performance')
    validate(val_loader, t_net, 0)

    for epoch in range(start_epoch, config.train_params.epochs + 1):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_with_distill(train_loader, d_net, optimizer, epoch)

        # evaluate on validation set
        err1, err5 = validate(val_loader, s_net, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': s_net.module.state_dict(),
            'train_state_dict': d_net.module.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        gc.collect()

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    step = 0
    while input is not None:
        # for PyTorch 0.4.x, volatile=True is replaced by with torch.no.grad(), so uncomment the followings:
        with torch.no_grad():
            output = model(input)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.train_params.print_freq == 0:
            print('Test (on val set): [Epoch {0}/{1}][Batch {2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, config.train_params.epochs, step, len(val_loader), batch_time=batch_time, top1=top1, top5=top5))
        input, target = prefetcher.next()
        step += 1

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'
          .format(epoch, config.train_params.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg


def train_with_distill(train_loader, d_net, optimizer, epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    btic = time.time()

    prefetcher = data_prefetcher(train_loader, is_sample=False)
    inputs, targets = prefetcher.next()

    step = 0
    while inputs is not None:

        batch_size = inputs.size(0)
        if step == 0:
            print('epoch %d lr %e' % (epoch, optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()

        outputs, loss = d_net(inputs, targets)

        loss = torch.mean(loss)
        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))

        train_loss.update(loss.item(), batch_size)
        top1.update(err1.item(), batch_size)
        top5.update(err5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.train_params.print_freq == 0:
            speed = config.train_params.print_freq * config.data.batch_size / (time.time() - btic)
            print(
                'Train with distillation: [Epoch %d/%d][Batch %d/%d]\t, speed %.3f, Loss %.3f, Top 1-error %.3f, Top 5-error %.3f' %
                (epoch, config.train_params.epochs, step, len(train_loader), speed, train_loss.avg, top1.avg, top5.avg))
            btic = time.time()

        inputs, targets = prefetcher.next()

        step += 1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "/job_data/%s/" % (config.net_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/job_data/%s/' % (config.net_type) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.optim.init_lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
