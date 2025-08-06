#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

torch::Tensor launch_swiglu_kernel(torch::Tensor& input);
torch::Tensor launch_swiglu_kernel_bwd(torch::Tensor& input, torch::Tensor& grad);


