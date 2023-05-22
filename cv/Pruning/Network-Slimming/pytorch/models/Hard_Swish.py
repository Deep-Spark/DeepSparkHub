import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
