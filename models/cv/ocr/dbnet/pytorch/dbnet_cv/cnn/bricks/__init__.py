# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from .activation import build_activation_layer

from .conv import build_conv_layer
from .conv_module import ConvModule

from .norm import build_norm_layer, is_norm
from .padding import build_padding_layer
from .plugin import build_plugin_layer
from .registry import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                       PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS)

from .upsample import build_upsample_layer
from .wrappers import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                       Linear, MaxPool2d, MaxPool3d)

