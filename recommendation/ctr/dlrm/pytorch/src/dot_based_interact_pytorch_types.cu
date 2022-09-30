#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include "dot_based_interact.cu"

torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input, torch::Tensor bottom_mlp_output, uint output_padding_width) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];
  uint output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols + output_padding_width;

  int64_t outputShape[2] = {batch_size, output_size};
  auto output = torch::empty(c10::IntArrayRef(outputShape), input.options());
  if (input.scalar_type() != torch::ScalarType::Half || bottom_mlp_output.scalar_type() != torch::ScalarType::Half) {
    throw std::invalid_argument("Invalid input type.");
  }
  dotBasedInteractFwd(input.contiguous().data_ptr<at::Half>(),
                      bottom_mlp_output.contiguous().data_ptr<at::Half>(),
                      output.contiguous().data_ptr<at::Half>(),
                      batch_size,
                      num_rows,
                      num_cols,
                      output_padding_width);
  return output;
}

std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input, torch::Tensor upstreamGrad, uint output_padding_width) {
  if (input.scalar_type() != torch::ScalarType::Half || upstreamGrad.scalar_type() != torch::ScalarType::Half) {
    throw std::invalid_argument("Invalid input type.");
  }
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];

  auto outputGrad = torch::empty_like(input);
  int64_t outputShape[2] = {batch_size, num_cols};
  auto mlp_grad = torch::empty(c10::IntArrayRef(outputShape), input.options());
  dotBasedInteractBwd(input.contiguous().data_ptr<at::Half>(),
                      upstreamGrad.contiguous().data_ptr<at::Half>(),
                      outputGrad.contiguous().data_ptr<at::Half>(),
                      mlp_grad.contiguous().data_ptr<at::Half>(),
                      batch_size,
                      num_rows,
                      num_cols,
                      output_padding_width);
  return {outputGrad, mlp_grad};
}
