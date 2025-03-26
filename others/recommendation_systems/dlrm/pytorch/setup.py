"""Simple setup script"""

import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

print(find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))

setup(name="dlrm",
      package_dir={'dlrm': 'dlrm'},
      version="1.0.0",
      description="Reimplementation of Facebook's DLRM",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=requirements,
      zip_safe=False,
      ext_modules=[
          CUDAExtension(name="dlrm.cuda_ext",
                        sources=[
                            os.path.join(abspath, "src/pytorch_ops.cpp"),
                            # os.path.join(abspath, "src/dot_based_interact_pytorch_types.cu"),
                            os.path.join(abspath, "src/gather_gpu.cu")
                        ],
                        extra_compile_args={
                            'cxx': [],
                            'nvcc' : [
                                '-DCUDA_HAS_FP16=1',
                                '-D__CUDA_NO_HALF_OPERATORS__',
                                '-D__CUDA_NO_HALF_CONVERSIONS__',
                                '-D__CUDA_NO_HALF2_OPERATORS__',
                                '-gencode', 'arch=compute_70,code=sm_70',
                                '-gencode', 'arch=compute_70,code=compute_70',
                                '-gencode', 'arch=compute_80,code=sm_80']
                        })
      ],
      cmdclass={"build_ext": BuildExtension})
