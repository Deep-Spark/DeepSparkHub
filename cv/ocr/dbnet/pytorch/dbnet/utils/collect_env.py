# Copyright (c) OpenMMLab. All rights reserved.
from dbnet_cv.utils import collect_env as collect_base_env
from dbnet_cv.utils import get_git_hash

import dbnet


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    # env_info['MMOCR'] = mmocr.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
