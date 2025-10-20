# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
from torch.nn import LayerNorm
from apex.normalization import FusedLayerNorm
import numpy as np
import torch

device = "cuda:0"

batch_size = 30
seq_len = 512
hidden_size = 1024

data = np.random.rand(batch_size,seq_len,hidden_size)
data = data*10 - 5 # -5,5

input1 = torch.tensor(data,dtype=torch.float32,device=device,requires_grad=True)
input2 = torch.tensor(data,dtype=torch.float32,device=device,requires_grad=True)


layer1 = LayerNorm(hidden_size).to(device)
layer2 = FusedLayerNorm(hidden_size).to(device)

output1 = layer1(input1)
output2 = layer2(input2)

diff = torch.abs(output1-output2)
print(f"forward diff max:{diff.max()}, diff min:{diff.min()}")
print("forward:",(diff<1e-6).all())

bk_data = torch.rand_like(output1)

output1.backward(bk_data.clone())
output2.backward(bk_data.clone())

grad_1 = input1.grad
grad_2 = input2.grad

diff = torch.abs(grad_1-grad_2)
print(f"backward diff max:{diff.max()}, diff min:{diff.min()}")
print("backward:",(diff<1e-6).all())
