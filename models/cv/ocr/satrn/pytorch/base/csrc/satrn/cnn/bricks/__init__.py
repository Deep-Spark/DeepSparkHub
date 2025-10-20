from .activation import build_activation_layer

from .conv import build_conv_layer
from .conv_module import ConvModule

from .drop import Dropout, DropPath

from .hsigmoid import HSigmoid
from .hswish import HSwish

from .norm import build_norm_layer, is_norm
from .padding import build_padding_layer

from .registry import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                       PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS)
from .scale import Scale
from .swish import Swish

from .wrappers import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                       Linear, MaxPool2d, MaxPool3d)


# __all__ = [
#     'ConvModule', 'build_activation_layer', 'build_conv_layer',
#     'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
#     'build_plugin_layer', 'is_norm', 'HSigmoid', 'HSwish',
#     'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS', 'PADDING_LAYERS',
#     'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale',
#     'Swish', 'Linear',
#     'Conv2d', 'ConvTranspose2d', 'MaxPool2d',
#     'ConvTranspose3d', 'MaxPool3d', 'Conv3d', 'Dropout', 'DropPath'
# ]

__all__ = [
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer',
    'is_norm', 'HSigmoid', 'HSwish',
    'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS', 'PADDING_LAYERS',
    'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale',
    'Swish', 'Linear',
    'Conv2d', 'ConvTranspose2d', 'MaxPool2d',
    'ConvTranspose3d', 'MaxPool3d', 'Conv3d', 'Dropout', 'DropPath'
]