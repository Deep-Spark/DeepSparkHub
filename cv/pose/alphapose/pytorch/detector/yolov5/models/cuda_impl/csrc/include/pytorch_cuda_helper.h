#ifndef CUDA_IMPL_COMMON_PYTORCH_CUDA_HELPER_H_
#define CUDA_IMPL_COMMON_PYTORCH_CUDA_HELPER_H_

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "cuda_helper.h"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // CUDA_IMPL_COMMON_PYTORCH_CUDA_HELPER_H_
