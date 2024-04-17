import spconv
from spconv.test_utils import params_grid, generate_sparse_data

import numpy as np
import torch

np.random.seed(484)
device = torch.device("cuda:0")

batchsize = 1

features = np.random.rand(1000,64).astype(np.float32)
features = torch.from_numpy(features)
indices = np.random.rand(1000,4).astype(np.int32)
indices = torch.from_numpy(indices)
shape = [19, 18, 17]

x = spconv.SparseConvTensor(features, indices, shape, batchsize)
x_dense_NCHW = x.dense()
print(x.sparity)