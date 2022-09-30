# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from resnet import ResNet, Bottleneck

__all__ = ['resnest14', 'resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}


class ResNeSt(ResNet):
    """ResNeSt

    Examples
    --------
    >>> resnest14 = ResNeSt(
    >>>     layers=[1, 1, 1, 1],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest26 = ResNeSt(
    >>>     layers=[2, 2, 2, 2],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest50 = ResNeSt(
    >>>     layers=[3, 4, 6, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest50_fast_4s2x40d = ResNeSt(
    >>>     layers=[3, 4, 6, 3],
    >>>     radix=4,
    >>>     groups=2,
    >>>     bottleneck_width=40,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=True)

    >>> resnest101 = ResNeSt(
    >>>     layers=[3, 4, 23, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest200 = ResNeSt(
    >>>     layers=[3, 24, 36, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest269 = ResNeSt(
    >>>     layers=[3, 30, 48, 8],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)
    """

    def __init__(self, **kwargs):
        super(ResNeSt, self).__init__(
            block=Bottleneck, **kwargs)


def resnest14(num_classes, **kwargs):
    model = ResNeSt(
        num_classes=num_classes,
        layers=[1, 1, 1, 1],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False)
    return model


def resnest26(num_classes, **kwargs):
    model = ResNeSt(
        num_classes=num_classes,
        layers=[2, 2, 2, 2],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False)
    return model


def resnest50(num_classes, **kwargs):
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = ResNeSt(
        num_classes=num_classes,
        layers=[3, 4, 6, 3],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    return model


def resnest101(num_classes, **kwargs):
    model = ResNeSt(
        num_classes=num_classes,
        layers=[3, 4, 23, 3],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    return model


def resnest200(num_classes, **kwargs):
    model = ResNeSt(
        num_classes=num_classes,
        layers=[3, 24, 36, 3],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    return model


def resnest269(num_classes, **kwargs):
    model = ResNeSt(
        num_classes=num_classes,
        layers=[3, 30, 48, 8],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    return model
