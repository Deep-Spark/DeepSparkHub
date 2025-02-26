import numpy as np
import torch
import torch.nn as nn

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.):
    """
    Label smoothing implementation.
    This function is taken from https://github.com/MIT-HAN-LAB/ProxylessNAS/blob/master/proxyless_nas/utils.py
    """

    logsoftmax = nn.LogSoftmax().cuda()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))