# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
from ...cuda_extension import _CudaExtension
from ...utils import append_nvcc_threads, get_cuda_cc_flag


class LayerNormCudaExtension(_CudaExtension):
    def __init__(self):
        super().__init__(name="layernorm_cuda")

    def sources_files(self):
        ret = [self.csrc_abs_path(fname) for fname in ["kernel/cuda/layer_norm_kernel.cu"]] + [
            self.pybind_abs_path("layernorm/layer_norm.cpp")
        ]
        return ret

    def include_dirs(self):
        ret = [self.get_cuda_home_include()] + [self.csrc_abs_path("")]
        return ret

    def cxx_flags(self):
        return ["-O3"] + self.version_dependent_macros

    def nvcc_flags(self):
        ret = ["-O3"] + self.version_dependent_macros + super().nvcc_flags()
        return append_nvcc_threads(ret)
