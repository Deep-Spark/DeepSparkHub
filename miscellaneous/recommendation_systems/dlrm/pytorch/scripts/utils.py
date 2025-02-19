"""utils originally from torchvision https://github.com/pytorch/vision/blob/master/references/classification/utils.py
With some changes.
    Removing iterable wrapper in MetricLogger
"""

from collections import defaultdict, deque
import datetime
from time import time
import torch
import torch.distributed as dist

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not dist.is_initialized() and dist.is_available():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def print(self, header=None):
        if not header:
            header = ''
        print_str = header
        for name, meter in self.meters.items():
            print_str += F"  {name}: {meter}"
        print(print_str)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class LearningRateScheduler:
    """Polynomial learning rate decay for multiple optimizers and multiple param groups

    Args:
        optimizers (list): optimizers for which to apply the learning rate changes
        base_lrs (list): a nested list of base_lrs to use for each param_group of each optimizer
        warmup_steps (int): number of linear warmup steps to perform at the beginning of training
        warmup_factor (int): warmup factor according to the MLPerf warmup rule
            see: https://github.com/mlperf/training_policies/blob/master/training_rules.adoc#91-hyperparameters-and-optimizer
            for more information
        decay_steps (int): number of steps over which to apply poly LR decay from base_lr to 0
        decay_start_step (int): the optimization step at which to start decaying the learning rate
            if None will start the decay immediately after
        decay_power (float): polynomial learning rate decay power
        end_lr_factor (float): for each optimizer and param group:
            lr = max(current_lr_factor, end_lr_factor) * base_lr

    Example:
        lr_scheduler = LearningRateScheduler(optimizers=[optimizer], base_lrs=[[lr]],
                                             warmup_steps=100, warmup_factor=0,
                                             decay_start_step=1000, decay_steps=2000,
                                             decay_power=2, end_lr_factor=1e-6)

        for batch in data_loader:
            lr_scheduler.step()
            # foward, backward, weight update
    """
    def __init__(self, optimizers, base_lrs, warmup_steps, warmup_factor,
                 decay_steps, decay_start_step, decay_power=2, end_lr_factor=0):
        self.current_step = 0
        self.optimizers = optimizers
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step
        self.decay_power = decay_power
        self.end_lr_factor = end_lr_factor
        self.decay_end_step = self.decay_start_step + self.decay_steps

        if self.decay_start_step < self.warmup_steps:
            raise ValueError('Learning rate warmup must finish before decay starts')

    def _compute_lr_factor(self):
        lr_factor = 1

        if self.current_step <= self.warmup_steps:
            warmup_step = 1 / (self.warmup_steps * (2 ** self.warmup_factor))
            lr_factor = 1 - (self.warmup_steps - self.current_step) * warmup_step
        elif self.decay_start_step < self.current_step <= self.decay_end_step:
            lr_factor = ((self.decay_end_step - self.current_step) / self.decay_steps) ** self.decay_power
            lr_factor = max(lr_factor, self.end_lr_factor)
        elif self.current_step > self.decay_end_step:
            lr_factor = self.end_lr_factor

        return lr_factor

    def step(self):
        self.current_step += 1
        lr_factor = self._compute_lr_factor()


        for optim, base_lrs in zip(self.optimizers, self.base_lrs):
            if isinstance(base_lrs, float):
                base_lrs = [base_lrs]
            for group_id, base_lr in enumerate(base_lrs):
                optim.param_groups[group_id]['lr'] = base_lr * lr_factor


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def timer_start():
    """Synchronize, start timer and profiler"""
    torch.cuda.profiler.start()
    torch.cuda.synchronize()
    start_time = time()
    return start_time

def timer_stop():
    """Synchronize, stop timer and profiler"""
    torch.cuda.synchronize()
    stop_time = time()
    torch.cuda.profiler.stop()
    return stop_time
