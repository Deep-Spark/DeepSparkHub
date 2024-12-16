#include "pack_type.cuh"
#include "swiglu.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

template<int pack_size, typename T>
__global__ void SwiGLUPackedKernel(const T* x, T* y, int pack_len, int last_dim) {
  int data_index = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_x = data_index % last_dim + (data_index / last_dim)  * last_dim * 2;
  int idx_y = idx_x + last_dim;
  const Packed<T, pack_size>* ptr_x = reinterpret_cast<const Packed<T, pack_size>*>(x);
  Packed<T, pack_size>* ptr_z = reinterpret_cast<Packed<T, pack_size>*>(y);
  Packed<T, pack_size> in_x, in_y, out;
  if (data_index < pack_len) {
    in_x = ptr_x[idx_x];
    in_y = ptr_x[idx_y];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      float x_f = PackItemType2Float(in_x.elem[i]);
      float y_f = PackItemType2Float(in_y.elem[i]);
      float z_f = x_f * y_f / (1.0f + ::exp(-x_f));
      out.elem[i] = Float2PackItemType<T>(z_f);
    }
    ptr_z[data_index] = out;
  }
}


template<typename T>
void LaunchSwiGLUPackedKernel(const T* x, T* y, const cudaStream_t &stream, int len, int last_dim) {
  constexpr const int pack_size = std::max(static_cast<int>(4 / sizeof(T)), 1);
  bool is_packed = IsAlignedForPack<pack_size, T, T>(x, y);
  if (is_packed && last_dim % pack_size == 0) {
    int pack_len = len / pack_size;
    unsigned int block_x = std::min(pack_len, 1024);
    unsigned int grid_x = (pack_len + block_x - 1) / block_x;
    SwiGLUPackedKernel<pack_size, T>
    <<<grid_x, block_x, 0, stream>>>(x, y, pack_len, last_dim / pack_size);
  } else {
    int pack_len = len;
    unsigned int block_x = std::min(pack_len, 1024);
    unsigned int grid_x = (pack_len + block_x - 1) / block_x;
    SwiGLUPackedKernel<1, T>
    <<<grid_x, block_x, 0, stream>>>(x, y, pack_len, last_dim);
  }
}


torch::Tensor launch_swiglu_kernel(torch::Tensor& input) {
  TORCH_CHECK(input.size(-1) % 2 == 0, "last dim of input should be even, got ", input.size(-1));
  auto shape = input.sizes().vec();
  shape[shape.size() - 1] = input.size(-1) / 2;
  torch::Tensor out = torch::empty(shape, at::TensorOptions(input.dtype())
                                              .device(input.device()));

  int last_dim = out.size(-1);
  int len = out.numel();
  const void* x = input.data_ptr();
  void* y = out.data_ptr();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input.dtype() == at::ScalarType::Half) {
    LaunchSwiGLUPackedKernel<__half>((const __half*)x, (__half*)y, stream, len, last_dim);
  } else if (input.dtype() == at::ScalarType::Float) {
    LaunchSwiGLUPackedKernel<float>((const float*)x, (float*)y, stream, len, last_dim);
  } else if (input.dtype() == at::ScalarType::BFloat16) {
    LaunchSwiGLUPackedKernel<__nv_bfloat16>((const __nv_bfloat16*)x, (__nv_bfloat16*)y, stream, len, last_dim);
  } else {
    TORCH_CHECK(false, "input datatype should be half/float, got ", input.dtype());
  }

  return out;
}
template<int pack_size, typename T>
__global__ void SwiGLUPackedBwdKernel(const T* x, const T* g, T* dx, int pack_len, int last_dim) {
  int data_index = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_x = data_index % last_dim + (data_index / last_dim)  * last_dim * 2;
  int idx_y = idx_x + last_dim;
  const Packed<T, pack_size>* ptr_x = reinterpret_cast<const Packed<T, pack_size>*>(x);
  const Packed<T, pack_size>* ptr_g = reinterpret_cast<const Packed<T, pack_size>*>(g);
  Packed<T, pack_size>* ptr_dx = reinterpret_cast<Packed<T, pack_size>*>(dx);
  Packed<T, pack_size> in_x, in_y, in_g, d_in_x, d_in_y;
  if (data_index < pack_len) {
    in_x = ptr_x[idx_x];
    in_y = ptr_x[idx_y];
    in_g = ptr_g[data_index];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      float x_f = PackItemType2Float(in_x.elem[i]);
      float y_f = PackItemType2Float(in_y.elem[i]);
      float g_f = PackItemType2Float(in_g.elem[i]);
      float x_sigmoid = 1.0f / (1.0f + ::exp(-x_f));
      float d_tmp_x = x_sigmoid * (1 + x_f * (1.0f - x_sigmoid)) * g_f * y_f;
      float d_tmp_y = x_f * x_sigmoid * g_f;
      d_in_x.elem[i] = Float2PackItemType<T>(d_tmp_x);
      d_in_y.elem[i] = Float2PackItemType<T>(d_tmp_y);
    }
    ptr_dx[idx_x] = d_in_x;
    ptr_dx[idx_y] = d_in_y;
  }
}


template<typename T>
void LaunchSwiGLUPackedBwdKernel(const T* x, const T* g, T* dx, const cudaStream_t &stream, int len, int last_dim) {
  constexpr const int pack_size = std::max(static_cast<int>(4 / sizeof(T)), 1);
  bool is_packed = IsAlignedForPack<pack_size, T, T, T>(x, g, dx);
  if (is_packed && last_dim % pack_size == 0) {
    int pack_len = len / pack_size;
    unsigned int block_x = std::min(pack_len, 1024);
    unsigned int grid_x = (pack_len + block_x - 1) / block_x;
    SwiGLUPackedBwdKernel<pack_size, T>
    <<<grid_x, block_x, 0, stream>>>(x, g, dx, pack_len, last_dim / pack_size);
  } else {
    int pack_len = len;
    unsigned int block_x = std::min(pack_len, 1024);
    unsigned int grid_x = (pack_len + block_x - 1) / block_x;
    SwiGLUPackedBwdKernel<1, T>
    <<<grid_x, block_x, 0, stream>>>(x, g, dx, pack_len, last_dim);
  }
}


torch::Tensor launch_swiglu_kernel_bwd(torch::Tensor& input, torch::Tensor& grad) {
  TORCH_CHECK(input.size(-1) % 2 == 0, "last dim of input should be even, got ", input.size(-1));
  int last_dim = grad.size(-1);
  int len = grad.numel();

  torch::Tensor out = torch::empty(input.sizes(), at::TensorOptions(input.dtype())
                                                     .device(input.device()));

  const void* x = input.data_ptr();
  const void* g = grad.data_ptr();
  void* dx = out.data_ptr();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (grad.dtype() == at::ScalarType::Half && input.dtype() == at::ScalarType::Half) {
    LaunchSwiGLUPackedBwdKernel<__half>((const __half*)x, (const __half*)g, (__half*)dx, stream, len, last_dim);
  } else if (grad.dtype() == at::ScalarType::Float && input.dtype() == at::ScalarType::Float) {
    LaunchSwiGLUPackedBwdKernel<float>((const float*)x, (const float*)g, (float*)dx, stream, len, last_dim);
  } else if (grad.dtype() == at::ScalarType::BFloat16 && input.dtype() == at::ScalarType::BFloat16) {
    LaunchSwiGLUPackedBwdKernel<__nv_bfloat16>((const __nv_bfloat16*)x, (const __nv_bfloat16*)g, (__nv_bfloat16*)dx, stream, len, last_dim);
  } else {
    TORCH_CHECK(false, "input and grad datatype should be half/float, got ", input.dtype(), grad.dtype());
  }

  return out;
}

