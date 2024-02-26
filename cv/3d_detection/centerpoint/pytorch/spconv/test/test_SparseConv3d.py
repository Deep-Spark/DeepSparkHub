from pathlib import Path
import spconv 
import torch
from torch import nn 
import numpy as np 
import time 
from spconv.test_utils import params_grid, generate_sparse_data, TestCase
import unittest

class SparseConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super().__init__()
        layers = [spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False)]
        for i in range(1, num_layers):
            layers.append(spconv.SparseConv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False))
        self.net = spconv.SparseSequential(
            *layers,
        )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors,self.shape, batch_size, self.grid)
        return self.net(x)# .dense()

class Conv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super().__init__()
        layers = [nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False)]
        for i in range(1, num_layers):
            layers.append(nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False))
        self.net = nn.Sequential(
            *layers,
        )
        self.shape = shape

    def forward(self, x):
        return self.net(x)# .dense()

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class TestSpConv(TestCase):
    
    def testSpConv3d(self):
        np.random.seed(484)
        devices = ["cuda:0"]
        shapes = [[30, 20, 10]]
        batchsizes = [1]

        in_channels = [64]
        out_channels = [32]
        ksizes = [2]
        strides = [1]
        paddings = [0]
        dilations = [1]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes, 
            strides, paddings, dilations):
            if all([s > 1, d > 1]):
                continue # don't support this.
            num_points = [1000] * bs
            device = torch.device(dev)

            sparse_dict = generate_sparse_data(shape, num_points, IC)
            #print(sparse_dict)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
            indices = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            # print(features.shape)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            # print(features_dense.shape)
            features_dense_t = torch.from_numpy(features_dense).to(device)
            # print(features_dense_t)
            features_dense_t.requires_grad = True
            net = SparseConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
            # print(net)
            net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
            filters_t = torch.from_numpy(filters).to(device)
            net_ref.net[0].weight.data[:] = filters_t.permute(4, 3, 0, 1, 2).contiguous()
            net.net[0].weight.data[:] = filters_t
            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs).dense()
            # print(out)
            dout = np.random.uniform(-0.2, 0.2, out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            out.backward(dout_t)
            print("backward end")
            print(np.linalg.norm(out.detach().cpu().numpy() - out_ref.detach().cpu().numpy()))
            # print(features_t.grad)
            # din_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4, 1).contiguous()
            # din_sparse = gather_nd(din_dense, indices_t.long())
            # din = features_t.grad.detach()
            # din_np = din.cpu().numpy()
            # din_sparse_np = din_sparse.cpu().numpy()

def main():
    test = TestSpConv()
    test.testSpConv3d()

if __name__ == '__main__':
    unittest.main()