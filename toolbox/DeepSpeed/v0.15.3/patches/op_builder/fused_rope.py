# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys
class FusedRopeBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ROPE"
    NAME = "fused_rope"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.rope.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/rope/fused_rotary_positional_embedding.cpp', 'csrc/rope/fused_rotary_positional_embedding_cuda.cu']

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []
    
    def include_paths(self):
        includes = ['csrc/includes']
        return includes
    def cxx_args(self):
        args = ['-O3']
        return args + self.version_dependent_macros()
    
    def nvcc_args(self):
        nvcc_flags = ['-O3','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__']  + self.version_dependent_macros()
        if self.is_rocm_pytorch():
            ROCM_MAJOR, ROCM_MINOR = self.installed_rocm_version()
            nvcc_flags += ['-DROCM_VERSION_MAJOR=%s' % ROCM_MAJOR, '-DROCM_VERSION_MINOR=%s' % ROCM_MINOR]
        else:
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo'] +
                self.compute_capability_args())
        return nvcc_flags

    
    
    
