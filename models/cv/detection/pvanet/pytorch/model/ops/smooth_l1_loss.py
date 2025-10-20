# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch


def smooth_l1_loss(input, target,
                   beta: float = 1. / 9,
                   size_average: bool = True,
                   reduction='sum'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average or reduction == 'average':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
