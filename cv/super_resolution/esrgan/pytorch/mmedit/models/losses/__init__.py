# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

from .gan_loss import DiscShiftLoss, GANLoss, GaussianBlur, GradientPenaltyLoss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss

