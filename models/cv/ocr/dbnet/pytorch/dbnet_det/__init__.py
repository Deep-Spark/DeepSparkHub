# Copyright (c) OpenMMLab. All rights reserved.
import dbnet_cv

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


dbnet_cv_minimum_version = '1.3.17'
dbnet_cv_maximum_version = '1.6.0'
dbnet_cv_version = digit_version(dbnet_cv.__version__)


assert (dbnet_cv_version >= digit_version(dbnet_cv_minimum_version)
        and dbnet_cv_version <= digit_version(dbnet_cv_maximum_version)), \
    f'DBNET_CV=={dbnet_cv.__version__} is used but incompatible. ' \
    f'Please install dbnet_cv>={dbnet_cv_minimum_version}, <={dbnet_cv_maximum_version}.'

__all__ = ['__version__', 'short_version']
