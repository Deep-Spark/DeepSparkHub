from pathlib import Path
import spconv 
import torch
from torch import nn 
import numpy as np 
import time 
from spconv.test_utils import params_grid, generate_sparse_data, TestCase
import unittest

class SparseCoupleDeConvTest(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size,
                 stride):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                indice_key="cp0",
                bias=False),
            spconv.SparseInverseConv2d(
                out_channels,
                in_channels,
                kernel_size,
                indice_key="cp0",
                bias=False),
            
        )
        self.todense = spconv.ToDense()
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors,self.shape, batch_size )
        return self.todense(self.net(x))# .dense()

def main():
    np.random.seed(484)
    devices = ["cuda:0"]
    shapes = [[20, 20]]
    batchsizes = [1]

    in_channels = [64]
    out_channels = [32]
    ksizes = [2]
    strides = [2]
    paddings = [0, 1]
    dilations = [1, 2]

    for dev, shape, bs, IC, OC, k, s in params_grid(
        devices, shapes, batchsizes, in_channels, out_channels, ksizes, 
        strides):
        device = torch.device(dev)
        num_points = [1000] * bs

        sparse_dict = generate_sparse_data(shape, num_points, IC)

        features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
        indices = np.ascontiguousarray(sparse_dict["indices"][:, [2, 0, 1]]).astype(np.int32)
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device)
        features_t = torch.from_numpy(features).to(device)
        features_t.requires_grad = True
        features_ref_t = torch.from_numpy(features).to(device)
        features_ref_t.requires_grad = True
        
        filters_0 = np.random.uniform(0, 1, size=[k, k, IC, OC]).astype(np.float32)
        filters_1 = np.random.uniform(0, 1, size=[k, k, OC, IC]).astype(np.float32)
        
        filters_t_0 = torch.from_numpy(filters_0).to(device)
        filters_t_1 = torch.from_numpy(filters_1).to(device)

        net = SparseCoupleDeConvTest(1, 2, shape, IC, OC, k, s).to(device)
        net.net[0].weight.data[:] = filters_t_0
        net.net[1].weight.data[:] = filters_t_1
        # print(net.net[0].weight.shape)
        out = net(features_t, indices_t, bs)
        print(out.detach().cpu().numpy())
        dout = np.random.uniform(-0.2, 0.2, out.shape).astype(features.dtype)
        dout_t = torch.from_numpy(dout).to(device)
        out.backward(dout_t)
        din = features_t.grad.detach()
        din_np = din.cpu().numpy()
        print(din_np)

if __name__ == '__main__':
    main()
    # unittest.main()