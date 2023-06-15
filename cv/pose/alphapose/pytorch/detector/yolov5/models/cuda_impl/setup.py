import glob
import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_sources():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(cur_dir, "csrc")
    sources = glob.glob(os.path.join(csrc_dir, "*.cpp"))
    cuda_sources = glob.glob(os.path.join(csrc_dir, "**", "*.cu"))

    if torch.cuda.is_available():
        sources += cuda_sources

    return sources


def get_include_dirs():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(cur_dir, "csrc")
    return [os.path.join(csrc_dir, "include")]


setup(
    name='cuda_impl',
    ext_modules=[
        CUDAExtension(
            name="cuda_impl.yolov5_decode",
            sources=get_sources(),
            include_dirs=get_include_dirs()
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    include_package_data=False,
    zip_safe=False
)
