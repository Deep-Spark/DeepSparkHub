# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
from __future__ import print_function
import os
import json
import argparse
import shutil
import torch
import torch.nn as nn
import models
import math
from apex import amp
from models.RanGer import Ranger
from models.Label_SmoothCELoss import LabelSmoothCELoss
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

dict={}
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
#parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--using-amp_loss', '-amp_loss', dest='amp_loss', action='store_true',
                    help='using amp_loss mix f16 and f32')
parser.add_argument('--s', type=float, default=1e-5,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate (default: 0.010)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')
parser.add_argument('--warm_up_epochs', default=5, type=int,
                    help='warm_up_epochs of the neural network')
parser.add_argument('--save', default='./logs/',
                    help='checkpoint__save')
parser.add_argument('--filename', default='',
                    help='filename__save')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

transform = [
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomRotation(45),
             transforms.RandomGrayscale(p=0.3),
             transforms.ColorJitter(brightness=1,contrast=1,saturation=0.5,hue=0.5),]

if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomChoice(transform),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.refine:
    args.refine = args.save + args.refine + '.pth'
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])

else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

#优化器
optimizer = Ranger(model.parameters(),lr=args.lr)  # 设置学习方法
warm_up_with_cosine_lr = lambda epoch: ((epoch+1) / args.warm_up_epochs) if (epoch+1) <= args.warm_up_epochs else 0.5 * (
                math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

#使用amp的量化训练
if args.amp_loss:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = data,target
        optimizer.zero_grad()
        output = model(data)
        criterion = LabelSmoothCELoss().cuda()
        loss = criterion(output, target)
        if args.amp_loss:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()


test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target
            output = model(data)
            criterion = LabelSmoothCELoss().cuda()
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best,filename,model_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_best)




def start():
    best_prec1 = 0.
    for epoch in range(0, args.epochs):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            lrr.append(param_group['lr'])
        train(epoch)
        prec1 = test(model)
        #保存最优的模型参数
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join(args.save, args.filename+'.pth'),
            os.path.join(args.save, args.filename+'_best.pth'))
        #这里只是保存网络的机构和通道数的配置，不包括权重参数。
        if epoch == 1:
            with open(os.path.join(args.save, args.filename+'.json'), 'w') as file_obj:
                for param_tensor in model.state_dict():
                    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                    dict[param_tensor] = model.state_dict()[param_tensor].size()
                    # dict = dict(dict)
                json.dump(dict, file_obj)

    print("Best accuracy: "+str(best_prec1))


if __name__ =="__main__":
    lrr=[]
    start()
    m = [x for x in range(args.epochs)]
    plt.plot(m,lrr)
    plt.savefig("cc.png")
    plt.show()




