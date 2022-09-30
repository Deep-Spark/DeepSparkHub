import glob
import os
import re
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension
EXT_TYPE = 'pytorch'
cmd_class = {'build_ext': BuildExtension}


def get_extensions():
    extensions = []

    if os.getenv('MMCV_WITH_OPS', '0') == '0':
        return extensions

    ext_name = 'sar._ext'
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))
    define_macros = []
    extra_compile_args = {'cxx': []}
    include_dirs = []

    is_rocm_pytorch = False
    try:
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                    (ROCM_HOME is not None)) else False
    except ImportError:
        pass

    project_dir = 'sar/ops/csrc/'
    if is_rocm_pytorch:
        from torch.utils.hipify import hipify_python

        hipify_python.hipify(
            project_directory=project_dir,
            output_directory=project_dir,
            includes='sar/ops/csrc/*',
            show_detailed=True,
            is_pytorch_extension=True,
        )
        define_macros += [('MMCV_WITH_CUDA', None)]
        define_macros += [('HIP_DIFF', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        op_files = glob.glob('./sar/ops/csrc/pytorch/hip/*')
        extension = CUDAExtension
        include_dirs.append(os.path.abspath('./sar/ops/csrc/common/hip'))
    elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('MMCV_WITH_CUDA', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        op_files = glob.glob('./sar/ops/csrc/pytorch/*.cpp') + \
            glob.glob('./sar/ops/csrc/pytorch/cuda/*.cu')
        extension = CUDAExtension
        include_dirs.append(os.path.abspath('./sar/ops/csrc/common'))
        include_dirs.append(os.path.abspath('./sar/ops/csrc/common/cuda'))
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./sar/ops/csrc/pytorch/*.cpp')
        extension = CppExtension
        include_dirs.append(os.path.abspath('./sar/ops/csrc/common'))

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)

    return extensions


setup(
    name='sar' if os.getenv('MMCV_WITH_OPS', '0') == '0' else 'sar-full',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False
)