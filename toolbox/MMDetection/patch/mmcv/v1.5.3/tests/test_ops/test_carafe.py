# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

import torch
from torch.autograd import gradcheck


class TestCarafe:

    def test_carafe_naive_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFENaive
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').float()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().float()
        gradcheck(CARAFENaive(5, 4, 2), (feat, mask), atol=1e-2, eps=1e-3)

    def test_carafe_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFE
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').float()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().float()
        gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-2, eps=1e-3)
