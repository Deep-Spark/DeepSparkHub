#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/ScalarType.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include "gn_kernel.h"
#include "Welford.h"
#include "vecs.h"
#define MAX_THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b

#define DEBUG_ENABLED 0
#if DEBUG_ENABLED
#define DEBUG(format, args...) fprintf(stderr, format, args)
#else
#define DEBUG(format, args...) ((void)0)
#endif
#define ELEM_DEBUG 0
#define INT int // torch uses int64_t but this came at a pretty big hit to performance and the input sizes that I frequently use (resolutions no bigger than 1024x1024) have a number of pixels smaller than the int max value

template <typename T>
struct acc_type { using type = float; };
template <>
struct acc_type<double> { using type = double; };

typedef struct block_params {
  int t; // threads per block
  int d; // dimensionality (number of rows of data that each threadblock proceesses in parallel)
  int f; // factor (number of different threadblocks needed to represent one row of data) 
} block_params_t;

inline block_params_t calc_block_params(const int ideal_num_threads, const int threads_per_row, int f_divides = -1, const int tpb_divides = -1) {
  /*
  ideal_num_threads: absolute upper limit of threads that a block should have (e.g. a kernel that operates on only 30 elements should have a max TPB of 30 (ideal_num_threads=30))
  threads_per_row: determines the user-specified upper limit on the size of blockDim.x
    - meant to be set to the size of the last dimension, e.g. a kernel operating on tensor sized (N, R, C) would have threads_per_row=C
  f_divides: optional parameter if user needs to explicitly specify a stricter requirement on the divisibility of the number of threads per block
    - e.g. fwd with C = 2560, G = 32, TPB = 480 wouldn't work since that means 32 groups are split over f=5 blocks (5.333 groups per block)
    - e.g. fwd with C = 2560, G = 32, TPB = 320 would work since that means 32 groups are split over f=8 blocks (4 groups per block), you could say that f divides 32 (f_divides=32)
  tpb_divides: optional parameter if user needs to explicitly specify that the returned threads per block needs to divide another value (e.g. a kernel where bounds checking isn't implemented)
    - e.g. fwd with H, W, C = 5, 5, 32; TPB = 512 wouldn't work since that means you use 1.5625 blocks to represent H*W*C (800) elements
    - e.g. fwd with H, W, C = 5, 5, 32; TPB = 160 would work since that means you use 5 blocks to represent H*W*C (800) elements, you could say that TPB (160) divides 800 (tpb_divides=800)
  */
  int TPB, d = 1, f = 1;
  f_divides = f_divides == -1 ? threads_per_row : f_divides;
  TPB = MIN(MAX_THREADS_PER_BLOCK, ideal_num_threads);
  if (threads_per_row < TPB) {
    d = TPB / threads_per_row;
    if (tpb_divides != -1) // could be put as another condition in the while loop but it hurts readability
      while (tpb_divides % (threads_per_row * d) != 0) // d = 1 guaranteed to break this condition
        --d;
  }
  else
    while (f_divides % f != 0 || threads_per_row / f > MAX_THREADS_PER_BLOCK)
      ++f;
  TPB = threads_per_row * d / f;
  return {TPB, d, f};
}

template <typename T> __device__ T inline identity(T x) {
  return x;
}
template <typename T> __device__ T inline identity_d(T /*x*/) {
  return 1;
}

template <typename T> __device__ T inline relu(T x) {
  return x > 0 ? x : static_cast<T>(0);
}
template <typename T> __device__ T inline relu_d(T x) {
  return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
}

template <typename T> __device__ T inline silu(T x) {
  return x / (1 + exp(-x));
}
template <typename T> __device__ T inline silu_d(T x) {
  const T s = 1 / (1 + exp(-x));
  return s * (1 + x * (1 - s));
}

template <typename T> __device__ T inline gelu(T x) {
  constexpr float kAlpha = M_SQRT1_2;
  return x * T(0.5) * (T(1) + erf(x * kAlpha));
}
template <typename T> __device__ T inline gelu_d(T x) {
  constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  constexpr float kAlpha = M_SQRT1_2;
  const T cdf = T(0.5) * (T(1) + erf(x * kAlpha));
  const T pdf = exp(T(-0.5) * x * x) * kBeta;
  return cdf + x * pdf;
}

template <typename T> __device__ T inline gelu_tanh(T x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715;
  auto inner = kBeta * (x + kKappa * x * x * x);
  return T(0.5) * x * (T(1) + tanh(inner));
}
template <typename T> __device__ T inline gelu_tanh_d(T x) {
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715;
  auto x_sq = x * x;
  auto x_cube = x_sq * x;
  auto inner = kBeta * (x + kKappa * x_cube);
  auto tanh_inner = tanh(inner);

  auto left = T(0.5) * x;
  auto right = T(1) + tanh_inner;

  auto left_derivative = T(0.5) * right;

  auto tanh_derivative = T(1) - tanh_inner * tanh_inner;
  auto inner_derivative = kBeta * (T(1) + T(3) * kKappa * x_sq);
  auto right_derivative = left * tanh_derivative * inner_derivative;

  return left_derivative + right_derivative;
}

//////////////////////////////////////////////////
// forward kernels
//////////////////////////////////////////////////

template <typename T>
__global__ void
compute_stats_pt1(
    const T* X,
    const int H,
    const int W,
    const int C,
    const int G,
    WelfordData<typename acc_type<T>::type, INT> *welford_data
  ) {
  /*
  Computes means and rstds of X on the W (width) dimension.
  grid: (x=N, y=H, z=f); block: (x=TPB/d, y=d)
  - TPB = Cd/f
  if TPB < C (f > 1, d=1)
    C = f*TPB
    X shape: (N, H, W, C) -view-> (N, H, W, 1, f, TPB); X stride: (HWC, WC, C, C, TPB, 1)
    dram reduction (per block): (W, 1, TPB) -reduce-> (1, TPB)
  else (block.x=C, block.y=d)
    TPB = Cd
    X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
    dram reduction (per block): (W/d, d, C) -reduce-> (d, C)
  shmem reduction (per block): (TPB,) -view-> (d, G/f, D) -permute-> (d, D, G/f) -reduce-> G/f
  output buffer: (N, f, G/f, H)
  */
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  const int w = ceil((float)W / d);
  int i;
#pragma unroll
  for (i = 0; i < w - 1; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C;
    reduce_idx += blockIdx.y * W * C;
    reduce_idx += i * d * C;
    reduce_idx += threadIdx.y * C;
    reduce_idx += blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x));
  }
  if ((int)(i * d + threadIdx.y) < W) // last iteration to deal with inputs with weird width sizes
    val = welford_op.reduce(val, static_cast<T_ACC>(X[blockIdx.x * H * W * C + blockIdx.y * W * C + i * d * C + threadIdx.y * C + blockIdx.z * TPB + threadIdx.x]));

  // shmem reduction
  const int D = C / G;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int f = gridDim.z;
  const int gf = G / f;
  const int d_idx = threadIdx.y;
  const int gf_idx = threadIdx.x / D;
  const int D_idx = threadIdx.x % D;

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  int idx = 0;
  idx += d_idx * D * gf;
  idx += D_idx * gf;
  idx += gf_idx;
  vals_reduced[idx] = val;
  __syncthreads();

  int reduce_n = TPB / gf; // number of inputs that gets reduced to a single output
#pragma unroll
  for (int stride = TPB / 2; stride >= gf && reduce_n % 2 == 0 && stride % gf == 0; stride >>= 1, reduce_n >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  if (tid < gf) {
#pragma unroll
    for (int i = 1; i < reduce_n; ++i)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + i * gf]);

    int out_idx = 0;
    out_idx += blockIdx.x * G * H;
    out_idx += blockIdx.z * gf * H;
    out_idx += tid * H;
    out_idx += blockIdx.y;
    welford_data[out_idx] = vals_reduced[tid];
  }
}

template <typename T>
__global__ void
compute_stats_pt2(
    WelfordData<typename acc_type<T>::type, INT> *welford_data,
    const int H,
    const int G,
    const T eps,
    T* means,
    T* rstds
  ) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, INT, thrust::pair<T_ACC, T_ACC>>;
  /*
  Computes means and rstds of X on the H (height) dimension.
  grid: (x=N, y=G); block: (x=H/f)
  - TPB = Gd/f
  welford_data shape: (N, G, H) -view-> (N, G, f, H/f); X stride: (GH, H, H/f, 1)
  dram reduction (per block): (f, H/f) -reduce-> (H/f,)
  shmem reduction (per block): (H/f) -reduce-> (1,)
  output buffer: (N, G)
  */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const int TPB = blockDim.y * blockDim.x;

  const int f = H / TPB;
  for (int i = 0 ; i < f; ++i) {
    int idx = 0;
    idx += blockIdx.x * G * H;
    idx += blockIdx.y * H;
    idx += i * H / f;
    idx += threadIdx.x;
    val = welford_op.combine(val, welford_data[idx]);
  }

  // shmem reduction
  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[MAX_THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int tid = threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();

  int reduce_n = TPB; // number of inputs that gets reduced to a single output

#pragma unroll
  for (int stride = TPB / 2; stride >= 1 && reduce_n % 2 == 0; stride >>= 1, reduce_n >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  if (tid == 0) {
#pragma unroll
    for (int i = 1; i < reduce_n; ++i)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + i]);

    T_ACC mean, var;
    thrust::tie(var, mean) = welford_op.project(vals_reduced[tid]);
    int out_idx = 0;
    out_idx += blockIdx.x * G;
    out_idx += blockIdx.y;
    means[out_idx] = mean;
    rstds[out_idx] = rsqrt(var + static_cast<T_ACC>(eps));
  }
}

template <typename T, int LOOP_I, int vec_elems, int64_t act_fn_option>
__global__ void
scale_shift(
    const T* X_data,
    const T* mean_data,
    const T* rstd_data,
    const T* weight_data,
    const T* bias_data,
    const int N,
    const int C,
    const int G,
    T* y
    ) {
  /*
  Performs elementwise op (X - mean) * rstd * weight + bias. Vectorized for speed.
  LOOP_I: number of elements that each thread processes.
  vec_elems: number of elements stored for each vector.
  grid: (x=NHWC / (TPB * LOOP_I * f), y=f), block: (x=TPB)
  - HWC % (TPB * LOOP_I * f) = 0
  - TPB * f % C = 0
  X shape: (N, H, W, C) -view-> (NHWC / (TPB * LOOP_I * f), LOOP_I, f, TPB); X.stride: (LOOP_I * f * TPB, f * TPB, TPB, 1)
  */
  using T_ACC = typename acc_type<T>::type;
  using V = float_vec<T, vec_elems>;
  const int f = gridDim.y;
  const int TPB = blockDim.x;

  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) % (C / vec_elems);
  const int g = (G * c) / (C / vec_elems);
  const int ng = n * G + g;
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  const V *weight_vecs = reinterpret_cast<const V*>(weight_data);
  const V *bias_vecs = reinterpret_cast<const V*>(bias_data);
  V *y_vecs = reinterpret_cast<V*>(y);
  T mean = mean_data[ng];
  T rstd = rstd_data[ng];
  V weight_vec = weight_vecs[c];
  V bias_vec = bias_vecs[c];

  // compute fused weight/bias a,b such that (x - mean) * rstd * weight + bias = x * a + b
  V fused_weight, fused_bias;
  if constexpr (vec_elems == 1) {
    fused_weight = {rstd * weight_vec.x};
    fused_bias = {-mean * fused_weight.x + bias_vec.x};
  }
  else if constexpr (vec_elems == 2) {
    fused_weight = {
      rstd * weight_vec.x,
      rstd * weight_vec.y
    };
    fused_bias = {
      -mean * fused_weight.x + bias_vec.x,
      -mean * fused_weight.y + bias_vec.y
    };
  }
  else if constexpr (vec_elems == 4) {
    fused_weight = {
      rstd * weight_vec.x,
      rstd * weight_vec.y,
      rstd * weight_vec.z,
      rstd * weight_vec.w
    };
    fused_bias = {
      -mean * fused_weight.x + bias_vec.x,
      -mean * fused_weight.y + bias_vec.y,
      -mean * fused_weight.z + bias_vec.z,
      -mean * fused_weight.w + bias_vec.w
    };
  }

  T (*act_fn)(T);
  if constexpr (act_fn_option == 0)
    act_fn = identity;
  else if constexpr (act_fn_option == 1)
    act_fn = relu;
  else if constexpr (act_fn_option == 2)
    act_fn = silu;
  else if constexpr (act_fn_option == 3)
    act_fn = gelu;
  else if constexpr (act_fn_option == 4)
    act_fn = gelu_tanh;

#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * f * TPB;
    idx += i * f * TPB;
    idx += blockIdx.y * TPB;
    idx += threadIdx.x;
    V X_vec = X_vecs[idx];
    
    if constexpr (vec_elems == 1)
      y_vecs[idx] = {act_fn(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x)};
    else if constexpr (vec_elems == 2) {
      y_vecs[idx] = {
        act_fn(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x),
        act_fn(static_cast<T_ACC>(X_vec.y) * fused_weight.y + fused_bias.y),
      };
    }
    else if constexpr (vec_elems == 4) {
      y_vecs[idx] = {
        act_fn(static_cast<T_ACC>(X_vec.x) * fused_weight.x + fused_bias.x),
        act_fn(static_cast<T_ACC>(X_vec.y) * fused_weight.y + fused_bias.y),
        act_fn(static_cast<T_ACC>(X_vec.z) * fused_weight.z + fused_bias.z),
        act_fn(static_cast<T_ACC>(X_vec.w) * fused_weight.w + fused_bias.w),
      };
    }
  }
}

template <typename T>
void run_gn_fwd_kernels(
    const T *X_data,
    const T *weight_data,
    const T *bias_data,
    const int N,
    const int H,
    const int W,
    const int C,
    const int G,
    T eps,
    const int64_t act_fn_option,
    T *Y_data,
    T *mean_data,
    T *rstd_data) {
  using T_ACC = typename acc_type<T>::type;
  using WelfordType = WelfordData<T_ACC, INT>;
  WelfordType *welford_data = (WelfordType*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(WelfordType) * N * G * H);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  
  // compute means/rstds over width dimension
  {
    auto [TPB, d, f] = calc_block_params(W * C, C, G);
    DEBUG("starting compute_stats 1, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", N, H, W, C, G, (C / G), TPB, d, f, (G / f));
    compute_stats_pt1<<<dim3(N, H, f), dim3(TPB / d, d), 0, cuda_stream>>>(
        X_data,
        H, W, C, G, 
        welford_data
    );
  }

  // compute means/rstds over height dimension
  {
    auto [TPB, d, f] = calc_block_params(H, H);
    DEBUG("starting compute_stats 2, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, d: %d, f: %d, G/f: %d\n", N, H, W, C, G, (C / G), TPB, d, f, (G / f));
    compute_stats_pt2<<<dim3(N, G), H / f, 0, cuda_stream>>>(
        welford_data,
        H, G, eps,
        mean_data, rstd_data
    );
  }

  // scale/shift X
  {
    const int D = C / G;
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;
    auto [TPB, d, f] = calc_block_params(H * W * C / 8 / vec_elems, C);

    if (!ELEM_DEBUG && ((H * W * C) % (TPB * 8 * f * vec_elems) == 0)) {
      const int LOOP_I = 8;
      const int num_blocks = N * H * W * C / TPB / LOOP_I / f;
      DEBUG("scale shift starting (LOOP_I = 8), N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, f: %d, num blocks (before vectors): %d, vec_elems: %d\n", N, H, W, C, G, D, TPB, f, num_blocks, vec_elems);
      if (vec_elems == 4 && act_fn_option == 0) // i'm sorry
        scale_shift<T, LOOP_I, 4, 0><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2 && act_fn_option == 0)
        scale_shift<T, LOOP_I, 2, 0><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 1 && act_fn_option == 0)
        scale_shift<T, LOOP_I, 1, 0><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 4 && act_fn_option == 1)
        scale_shift<T, LOOP_I, 4, 1><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2 && act_fn_option == 1)
        scale_shift<T, LOOP_I, 2, 1><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 1 && act_fn_option == 1)
        scale_shift<T, LOOP_I, 1, 1><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 4 && act_fn_option == 2)
        scale_shift<T, LOOP_I, 4, 2><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2 && act_fn_option == 2)
        scale_shift<T, LOOP_I, 2, 2><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 1 && act_fn_option == 2)
        scale_shift<T, LOOP_I, 1, 2><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 4 && act_fn_option == 3)
        scale_shift<T, LOOP_I, 4, 3><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2 && act_fn_option == 3)
        scale_shift<T, LOOP_I, 2, 3><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 1 && act_fn_option == 3)
        scale_shift<T, LOOP_I, 1, 3><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 4 && act_fn_option == 4)
        scale_shift<T, LOOP_I, 4, 4><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 2 && act_fn_option == 4)
        scale_shift<T, LOOP_I, 2, 4><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      else if (vec_elems == 1 && act_fn_option == 4)
        scale_shift<T, LOOP_I, 1, 4><<<dim3(num_blocks / vec_elems, f), TPB, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    }
    else {// relatively slow fallback
      const int num_blocks = N * H * W;
      DEBUG("SLOW FALLBACK, scale shift kernel starting, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, f: %d, num blocks (before vectors): %d, vec_elems: %d\n", N, H, W, C, G, D, C/f, f, num_blocks, vec_elems);
      if (act_fn_option == 0)
        scale_shift<T, 1, 1, 0><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      if (act_fn_option == 1)
        scale_shift<T, 1, 1, 1><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      if (act_fn_option == 2)
        scale_shift<T, 1, 1, 2><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      if (act_fn_option == 3)
        scale_shift<T, 1, 1, 3><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
      if (act_fn_option == 4)
        scale_shift<T, 1, 1, 4><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(X_data, mean_data, rstd_data, weight_data, bias_data, N, C, G, Y_data);
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(welford_data);
}

template void run_gn_fwd_kernels<float>(const float *X_data, const float *weight_data, const float *bias_data, const int N, const int h, const int W, const int C, const int G, float eps, const int64_t act_fn_option, float *Y_data, float *mean_data, float *rstd_data);
template void run_gn_fwd_kernels<double>(const double *X_data, const double *weight_data, const double *bias_data, const int N, const int h, const int W, const int C, const int G, double eps, const int64_t act_fn_option, double *Y_data, double *mean_data, double *rstd_data);
template void run_gn_fwd_kernels<c10::Half>(const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *bias_data, const int N, const int h, const int W, const int C, const int G, c10::Half eps, const int64_t act_fn_option, c10::Half *Y_data, c10::Half *mean_data, c10::Half *rstd_data);
template void run_gn_fwd_kernels<c10::BFloat16>(const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *bias_data, const int N, const int h, const int W, const int C, const int G, c10::BFloat16 eps, const int64_t act_fn_option, c10::BFloat16 *Y_data, c10::BFloat16 *mean_data, c10::BFloat16 *rstd_data);

//////////////////////////////////////////////////
// backward kernels
//////////////////////////////////////////////////

template <typename T>
__device__ void
sum_reduce(
    T vals_reduced,
    const int start_stride,
    const int end_stride
  ) {
  // Sums a shared buffer (vals_reduced) with shape (2 * start_stride / end_stride, end_stride) into (end_stride,).
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int reduce_n = 2 * start_stride / end_stride;

#pragma unroll
  for (int stride = start_stride; stride >= end_stride && reduce_n % 2 == 0 && stride % end_stride == 0; stride >>= 1, reduce_n >>= 1) {
    if (tid < stride)
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
  }

  if (tid < end_stride)
#pragma unroll
    for (int i = 1; i < reduce_n; ++i)
      vals_reduced[tid] += vals_reduced[tid + i * end_stride];
  __syncthreads();
}

template <typename T, int64_t act_fn_option>
__global__ void
width_reduce(
      const T* dy_data,
      const T* X_data,
      const T* mean_data,
      const T* rstd_data,
      const T* weight_data,
      const T* bias_data,
      const int H,
      const int W,
      const int C,
      const int G,
      typename acc_type<T>::type *xdy_dy_sum_data) {
  /*
  Loops over W (width) dimension, loading and summing dy, X, and the activation derivative of Y. Outputs stored in xdy_dy_sum_data. Spatial dimension H is processed in a separate kernel.
  grid: (x=N, y=H, z=f); blockdim: (x=TPB/d, y=d)
    TPB = Cd/f
  if TPB < C (f > 1, d=1)
    C = f*TPB
    X shape: (N, H, W, C) -view-> (N, H, W, 1, f, TPB); X stride: (HWC, WC, C, C, TPB, 1)
    dram reduction (per block): (W, 1, TPB) -reduce-> (TPB,)
  else (block.x=C, block.y=d)
    TPB = Cd
    X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
    dram reduction (per block): (W/d, d, C) -reduce-> (d, C)
  shmem reduction (per block): (TPB, 2) -> (d, C/f, 2) -reduce-> (C/f, 2) (the 2 comes from storing both xdy_sum and dy_sum in the same buffer)
  output buffer: (N, f, C/f, H, 2) -view-> (N, C, H, 2)
    xdy_dy_sum_data[:, :, :, 0] = x * dy * activation_derivative((x-mean)*rstd*weight+bias)
    xdy_dy_sum_data[:, :, :, 1] = dy * activation_derivative((x-mean)*rstd*weight+bias)
   */
  using T_ACC = typename acc_type<T>::type;

  const int TPB = blockDim.y * blockDim.x;
  const int d = blockDim.y;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;

  const int n = blockIdx.x;
  int c = blockIdx.z * blockDim.x + threadIdx.x;
  int g = G * c / C;
  const int ng = n * G + g;
  T_ACC fused_scale = rstd_data[ng] * weight_data[c];
  T_ACC fused_bias = -mean_data[ng] * fused_scale + bias_data[c];

  T (*act_d_fn)(T x);
  if constexpr (act_fn_option == 0)
    act_d_fn = identity_d;
  else if constexpr (act_fn_option == 1)
    act_d_fn = relu_d;
  else if constexpr (act_fn_option == 2)
    act_d_fn = silu_d;
  else if constexpr (act_fn_option == 3)
    act_d_fn = gelu_d;
  else if constexpr (act_fn_option == 4)
    act_d_fn = gelu_tanh_d;

  const int w = ceil((float)W / d);
  int i;
#pragma unroll
  for (i = 0; i < w - 1; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C;
    reduce_idx += blockIdx.y * W * C;
    reduce_idx += i * d * C;
    reduce_idx += threadIdx.y * C;
    reduce_idx += blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    T_ACC X_elem = static_cast<T_ACC>(X_data[reduce_idx]);
    T_ACC X_norm = X_elem * fused_scale + fused_bias;
    T_ACC d_act = act_d_fn(X_norm);
    xdy_sum += dy_elem * X_elem * d_act;
    dy_sum += dy_elem * d_act;
  }
  if ((int)(i * d + threadIdx.y) < W) { // last iteration to deal with inputs with weird width sizes
    int reduce_idx = blockIdx.x * H * W * C + blockIdx.y * W * C + i * d * C + threadIdx.y * C + blockIdx.z * TPB + threadIdx.x;
    T_ACC dy_elem = static_cast<T_ACC>(dy_data[reduce_idx]);
    T_ACC X_elem = static_cast<T_ACC>(X_data[reduce_idx]);
    T_ACC X_norm = X_elem * fused_scale + fused_bias;
    T_ACC d_act = act_d_fn(X_norm);
    xdy_sum += dy_elem * X_elem * d_act;
    dy_sum += dy_elem * d_act;
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[2 * tid] = xdy_sum;
  vals_reduced[2 * tid + 1] = dy_sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB, 2 * C);

  // put reduced outputs into return buffers
  if (tid < C) {
    int out_idx = 0;
    out_idx += blockIdx.x * C * H;
    out_idx += (blockIdx.z * TPB + tid) * H;
    out_idx += blockIdx.y;

    xdy_dy_sum_data[2 * out_idx] = vals_reduced[2 * tid];
    xdy_dy_sum_data[2 * out_idx + 1] = vals_reduced[2 * tid + 1];
  }
}

template <typename T>
__global__ void
height_reduce(
    T *xdy_dy_sum_data, // no need to specify T_ACC as T is already an accumulation type
    const int H,
    const int C,
    T *xdy_sum_data,
    T *dy_sum_data
  ) {
  /*
  Same thing as width_reduce but over the H (height) instead of the width dimension.
  grid: (x=N, y=C); block: (x=2H/f)
  X shape: (N, C, H, 2) -view-> (N, C, f, H/f, 2); X stride: (2CH, 2H, 2H/f, H/f, 1)
  dram reduction (per block): (f, H/f, 2) -reduce-> (H/f, 2)
  shmem reduction (per block): (H/f, 2) -reduce-> (2,)
  output buffer: (N, C, 2)
  */
  const int TPB = blockDim.x;
  const int tid = threadIdx.x;

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[];
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);
  T sum = 0;
  int i;
#pragma unroll
  for (i = 0; i < ceil((float)2 * H / TPB) - 1; ++i) {
    int idx = 0;
    idx += blockIdx.x * C * H * 2;
    idx += blockIdx.y * H * 2;
    idx += i * TPB;
    idx += tid;
    sum += xdy_dy_sum_data[idx];
  }
  if (i * TPB + tid < 2 * H)
    sum += xdy_dy_sum_data[blockIdx.x * C * H * 2 + blockIdx.y * H * 2 + i * TPB + tid];

  vals_reduced[tid] = sum;
  __syncthreads();
  sum_reduce(vals_reduced, TPB / 2, 2);

  // put reduced outputs into return buffers
  if (tid == 0) {
    int out_idx = blockIdx.x * C + blockIdx.y;
    xdy_sum_data[out_idx] = vals_reduced[0];
    dy_sum_data[out_idx] = vals_reduced[1];
  }
}

template <typename T>
__global__ void
compute_bwd_scale_biases(
    const T* mean_data,
    const T* rstd_data,
    const T* weight_data,
    const T* bias_data,
    typename acc_type<T>::type* xdy_sum_data,
    typename acc_type<T>::type* dy_sum_data,
    const int H,
    const int W,
    const int C,
    const int G,
    typename acc_type<T>::type* coef1_data,
    typename acc_type<T>::type* coef2_data,
    typename acc_type<T>::type* coef3_data,
    typename acc_type<T>::type* coef4_data
    ) {
  /*
  Calculates coefficients to reduce computation on the elementwise kernel.
  - coef1: fused scale (rstd * weight)
  - coef2: fused bias (-mean * rstd * weight + bias)
  - coef3/4: some derivative terms
  griddim: (x=N, y=f); blockdim: (x=C/f)
  - d = num. spatial elements (from HW dimension) each thread-block processes in parallel
  - Cd = TPB (threads per block)
  X shape: (N, C) -view-> (N, G, D) -permute-> (N, D, G) -reduce-> (N, G)
  shmem reduction: (D, G) -reduce-> G
  output buffer: (N, G)
  */
  using T_ACC = typename acc_type<T>::type;
  const int D = C / G;
  const int f = gridDim.y;
  const int Gf = G / f;
  const int n = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int g = c / D;
  const int d = c % D;
  const int nc = n * C + c;
  const T_ACC gamma_v = static_cast<T_ACC>(weight_data[c]);

  extern __shared__ char vals_reduced_uncasted[]; // size 2*C, C for sum1, C for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  int idx = 0;
  idx += d * G / f;
  idx += g % Gf;
  vals_reduced[2 * idx] = xdy_sum_data[nc] * gamma_v;
  vals_reduced[2 * idx + 1] = dy_sum_data[nc] * gamma_v;
  __syncthreads();
  sum_reduce(vals_reduced, C / f, 2 * G / f);

  const int ng = n * G + g;
  const T_ACC mean_elem = static_cast<T_ACC>(mean_data[ng]);
  const T_ACC rstd_elem = static_cast<T_ACC>(rstd_data[ng]);
  coef1_data[nc] = rstd_elem * weight_data[c];
  coef2_data[nc] = -mean_elem * rstd_elem * weight_data[c] + bias_data[c];

  if (d == 0) {
    const T_ACC sum1 = vals_reduced[2 * (g % Gf)];
    const T_ACC sum2 = vals_reduced[2 * (g % Gf) + 1];
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * H * W);
    const T_ACC x = (sum2 * mean_elem - sum1) * rstd_elem * rstd_elem * rstd_elem * s;
    coef3_data[ng] = x;
    coef4_data[ng] = (-x * mean_elem) - (sum2 * s * rstd_elem);
  }
}

template <typename T>
__global__ void
compute_dweight_dbias(
    const T* mean_data,
    const T* rstd_data,
    typename acc_type<T>::type *xdy_sum_data,
    typename acc_type<T>::type *dy_sum_data,
    const int N,
    const int C,
    const int G,
    T* dweight_data,
    T* dbias_data) {
  /*
  Computes derivatives wrt the weight and bias. 
  grid: (x=f), block: (x=C/f)
  */
  using T_ACC = typename acc_type<T>::type;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int D = C / G;
  const int g = c / D;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;

#pragma unroll
  for (int n = 0; n < N; ++n) {
    const int nc = n * C + c;
    const int ng = n * G + g;
    sum1 += (xdy_sum_data[nc] - dy_sum_data[nc] * mean_data[ng]) * rstd_data[ng];
    sum2 += dy_sum_data[nc];
  }
  dweight_data[c] = sum1;
  dbias_data[c] = sum2;
}

template <typename T, int LOOP_I, int vec_elems, int64_t act_fn_option>
__global__ void
dx_elem_kernel(
    const T* dy_data,
    const T* X_data,
    typename acc_type<T>::type* coef1_data,
    typename acc_type<T>::type* coef2_data,
    typename acc_type<T>::type* coef3_data,
    typename acc_type<T>::type* coef4_data,
    const int N,
    const int C,
    const int G,
    T* dx_data
    ) {
  /*
  Performs elementwise kernel to calculate gradients wrt X. Vectorized for speed.
  LOOP_I: number of elements that each thread processes.
  vec_elems: number of elements stored for each vector.
  grid: (x=NHWC / (TPB * LOOP_I * f), y=f), block: (x=TPB)
  - HWC % (TPB * LOOP_I * f) = 0
  - TPB * f % C = 0
  X shape: (N, H, W, C) -view-> (NHWC / (TPB * LOOP_I * f), LOOP_I, f, TPB); X.stride: (LOOP_I * f * TPB, f * TPB, TPB, 1)
  */
  using T_ACC = typename acc_type<T>::type;
  using V = float_vec<T, vec_elems>;
  using V_ACC = float_vec<T_ACC, vec_elems>;
  const int f = gridDim.y;
  const int n = (N * blockIdx.x) / gridDim.x;
  const int c = (blockIdx.y * blockDim.x + threadIdx.x) % (C / vec_elems);
  const int g = (G * c) / (C / vec_elems);
  const int nc = n * (C / vec_elems) + c;
  const int ng = n * G + g;
  T_ACC coef3 = coef3_data[ng];
  T_ACC coef4 = coef4_data[ng];
  const V *dy_vecs = reinterpret_cast<const V*>(dy_data);
  const V *X_vecs = reinterpret_cast<const V*>(X_data);
  V *dx_vecs = reinterpret_cast<V*>(dx_data);
  V_ACC coef1_vec = reinterpret_cast<V_ACC*>(coef1_data)[nc];
  V_ACC coef2_vec = reinterpret_cast<V_ACC*>(coef2_data)[nc];

  T (*act_d_fn)(T);
  if constexpr (act_fn_option == 0)
    act_d_fn = identity_d;
  else if constexpr (act_fn_option == 1)
    act_d_fn = relu_d;
  else if constexpr (act_fn_option == 2)
    act_d_fn = silu_d;
  else if constexpr (act_fn_option == 3)
    act_d_fn = gelu_d;
  else if constexpr (act_fn_option == 4)
    act_d_fn = gelu_tanh_d;

#pragma unroll
  for (int i = 0; i < LOOP_I; ++i) {
    int idx = 0;
    idx += blockIdx.x * LOOP_I * f * blockDim.x;
    idx += i * f * blockDim.x;
    idx += blockIdx.y * blockDim.x;
    idx += threadIdx.x;

    V dy_vec = dy_vecs[idx];
    V X_vec = X_vecs[idx];

    if constexpr (vec_elems == 1) {
      V X_norm = {X_vec.x * coef1_vec.x + coef2_vec.x};
      dx_vecs[idx] = {
        (coef1_vec.x * act_d_fn(X_norm.x) * dy_vec.x)
          + ((coef3 * X_vec.x) + coef4)
      };
    }
    else if constexpr (vec_elems == 2) {
      V X_norm = {
        X_vec.x * coef1_vec.x + coef2_vec.x,
        X_vec.y * coef1_vec.y + coef2_vec.y,
      };
      dx_vecs[idx] = {
        (coef1_vec.x * act_d_fn(X_norm.x) * dy_vec.x)
          + ((coef3 * X_vec.x) + coef4),
        (coef1_vec.y * act_d_fn(X_norm.y) * dy_vec.y)
          + ((coef3 * X_vec.y) + coef4),
      };
    }
    else if constexpr (vec_elems == 4) {
      V X_norm = {
        X_vec.x * coef1_vec.x + coef2_vec.x,
        X_vec.y * coef1_vec.y + coef2_vec.y,
        X_vec.z * coef1_vec.z + coef2_vec.z,
        X_vec.w * coef1_vec.w + coef2_vec.w,
      };
      dx_vecs[idx] = {
        (coef1_vec.x * act_d_fn(X_norm.x) * dy_vec.x)
          + ((coef3 * X_vec.x) + coef4),
        (coef1_vec.y * act_d_fn(X_norm.y) * dy_vec.y)
          + ((coef3 * X_vec.y) + coef4),
        (coef1_vec.z * act_d_fn(X_norm.z) * dy_vec.z)
          + ((coef3 * X_vec.z) + coef4),
        (coef1_vec.w * act_d_fn(X_norm.w) * dy_vec.w)
          + ((coef3 * X_vec.w) + coef4),
      };
    }
  }
}

template <typename T>
void run_gn_bwd_kernels(
      const T *dy_data,
      const T *X_data,
      const T *weight_data,
      const T *bias_data,
      const T *mean_data,
      const T *rstd_data,
      const int N,
      const int H,
      const int W,
      const int C,
      const int G,
      const int64_t act_fn_option,
      T *dx_data,
      T *dweight_data,
      T *dbias_data
  ) {
  using T_ACC = typename acc_type<T>::type;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int D = C / G;

  T_ACC* xdy_dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C * H * 2);

  // sum over W dim
  {
    auto [TPB, d, f] = calc_block_params(W * C, C, G);
    DEBUG("starting width reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    if (act_fn_option == 0)
      width_reduce<T, 0><<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB, cuda_stream>>>(
          dy_data, X_data, 
          mean_data, rstd_data,
          weight_data, bias_data,
          H, W, C, G,
          xdy_dy_sum_data);
    else if (act_fn_option == 1)
      width_reduce<T, 1><<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, xdy_dy_sum_data);
    else if (act_fn_option == 2)
      width_reduce<T, 2><<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, xdy_dy_sum_data);
    else if (act_fn_option == 3)
      width_reduce<T, 3><<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, xdy_dy_sum_data);
    else if (act_fn_option == 4)
      width_reduce<T, 4><<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2 * TPB, cuda_stream>>>(dy_data, X_data, mean_data, rstd_data, weight_data, bias_data, H, W, C, G, xdy_dy_sum_data);
  }

  T_ACC* xdy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC* dy_sum_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  // sum over H dim
  {
    auto [TPB, d, f] = calc_block_params(2 * H, 2);
    DEBUG("starting height reduce, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    height_reduce<<<dim3(N, C), TPB, sizeof(T_ACC) * TPB, cuda_stream>>>(
        xdy_dy_sum_data,
        H, C,
        xdy_sum_data, dy_sum_data);
  }
  c10::cuda::CUDACachingAllocator::raw_delete(xdy_dy_sum_data);

  // compute weight/bias grads
  {
    auto [TPB, d, f] = calc_block_params(C, C, G);
    DEBUG("starting compute dweight dbias, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    compute_dweight_dbias<<<f, C / f, 0, cuda_stream>>>(
        mean_data, rstd_data,
        xdy_sum_data, dy_sum_data,
        N, C, G,
        dweight_data, dbias_data);
  }

  T_ACC *coef1_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC *coef2_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * C);
  T_ACC *coef3_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);
  T_ACC *coef4_data = (T_ACC*)c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(T_ACC) * N * G);
  // compute fused scales/biases for dx elementwise kernel
  {
    auto [TPB, d, f] = calc_block_params(C, C, G);
    DEBUG("starting bwd scale biases, N: %d, H: %d, W: %d, C: %d, G: %d, TPB: %d, d: %d, f: %d\n", N, H, W, C, G, TPB, d, f);
    compute_bwd_scale_biases<<<dim3(N, f), C / f, sizeof(T_ACC) * 2 * C / f, cuda_stream>>>(
        mean_data, rstd_data, weight_data, bias_data,
        xdy_sum_data, dy_sum_data,
        H, W, C, G,
        coef1_data, coef2_data, coef3_data, coef4_data);
  }

  {
    int vec_elems;
    if (D % 4 == 0) vec_elems = 4;
    else if (D % 2 == 0) vec_elems = 2;
    else vec_elems = 1;
    auto [TPB, d, f] = calc_block_params(H * W * C, C, G);

    if (!ELEM_DEBUG && ((H * W * C) % (TPB * 8 * f * vec_elems) == 0)) {
      const int LOOP_I = 8;
      const int num_blocks = ceil((float)N * H * W * C / TPB / LOOP_I / f);
      DEBUG("dx elem kernel starting, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, f: %d, num blocks (before vectors): %d, vec_elems: %d\n", N, H, W, C, G, D, TPB, f, num_blocks, vec_elems);
      if (D % 4 == 0 && act_fn_option == 0)
        dx_elem_kernel<T, LOOP_I, 4, 0><<<dim3(num_blocks / 4, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 2 == 0 && act_fn_option == 0)
        dx_elem_kernel<T, LOOP_I, 2, 0><<<dim3(num_blocks / 2, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 1 == 0 && act_fn_option == 0)
        dx_elem_kernel<T, LOOP_I, 1, 0><<<dim3(num_blocks / 1, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 4 == 0 && act_fn_option == 1)
        dx_elem_kernel<T, LOOP_I, 4, 1><<<dim3(num_blocks / 4, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 2 == 0 && act_fn_option == 1)
        dx_elem_kernel<T, LOOP_I, 2, 1><<<dim3(num_blocks / 2, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 1 == 0 && act_fn_option == 1)
        dx_elem_kernel<T, LOOP_I, 1, 1><<<dim3(num_blocks / 1, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 4 == 0 && act_fn_option == 2)
        dx_elem_kernel<T, LOOP_I, 4, 2><<<dim3(num_blocks / 4, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 2 == 0 && act_fn_option == 2)
        dx_elem_kernel<T, LOOP_I, 2, 2><<<dim3(num_blocks / 2, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 1 == 0 && act_fn_option == 2)
        dx_elem_kernel<T, LOOP_I, 1, 2><<<dim3(num_blocks / 1, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 4 == 0 && act_fn_option == 3)
        dx_elem_kernel<T, LOOP_I, 4, 3><<<dim3(num_blocks / 4, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 2 == 0 && act_fn_option == 3)
        dx_elem_kernel<T, LOOP_I, 2, 3><<<dim3(num_blocks / 2, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 1 == 0 && act_fn_option == 3)
        dx_elem_kernel<T, LOOP_I, 1, 3><<<dim3(num_blocks / 1, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 4 == 0 && act_fn_option == 4)
        dx_elem_kernel<T, LOOP_I, 4, 4><<<dim3(num_blocks / 4, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 2 == 0 && act_fn_option == 4)
        dx_elem_kernel<T, LOOP_I, 2, 4><<<dim3(num_blocks / 2, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (D % 1 == 0 && act_fn_option == 4)
        dx_elem_kernel<T, LOOP_I, 1, 4><<<dim3(num_blocks / 1, f), TPB, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
    }
    else { // relatively slow fallback
      const int num_blocks = N * H * W;
      DEBUG("SLOW FALLBACK, dx elem kernel starting, N: %d, H: %d, W: %d, C: %d, G: %d, D: %d, TPB: %d, f: %d, num blocks (before vectors): %d, vec_elems: %d\n", N, H, W, C, G, D, C/f, f, num_blocks, vec_elems);
      if (act_fn_option == 0)
        dx_elem_kernel<T, 1, 1, 0><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (act_fn_option == 1)
        dx_elem_kernel<T, 1, 1, 1><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (act_fn_option == 2)
        dx_elem_kernel<T, 1, 1, 2><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (act_fn_option == 3)
        dx_elem_kernel<T, 1, 1, 3><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
      else if (act_fn_option == 4)
        dx_elem_kernel<T, 1, 1, 4><<<dim3(num_blocks, f), C / f, 0, cuda_stream>>>(dy_data, X_data, coef1_data, coef2_data, coef3_data, coef4_data, N,  C, G, dx_data);
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(xdy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(dy_sum_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef1_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef2_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef3_data);
  c10::cuda::CUDACachingAllocator::raw_delete(coef4_data);
}

template void run_gn_bwd_kernels<double>(const double *dy_data, const double *X_data, const double *weight_data, const double *bias_data, const double *mean_data, const double *rstd_data, const int N, const int H, const int W, const int C, const int G, const int64_t act_fn_option, double *dx_data, double *dweight_data, double *dbias_data);
template void run_gn_bwd_kernels<float>(const float *dy_data, const float *X_data, const float *weight_data, const float *bias_data, const float *mean_data, const float *rstd_data, const int N, const int H, const int W, const int C, const int G, const int64_t act_fn_option, float *dx_data, float *dweight_data, float *dbias_data);
template void run_gn_bwd_kernels<c10::Half>(const c10::Half *dy_data, const c10::Half *X_data, const c10::Half *weight_data, const c10::Half *bias_data, const c10::Half *mean_data, const c10::Half *rstd_data, const int N, const int H, const int W, const int C, const int G, const int64_t act_fn_option, c10::Half *dx_data, c10::Half *dweight_data, c10::Half *dbias_data);
template void run_gn_bwd_kernels<c10::BFloat16>(const c10::BFloat16 *dy_data, const c10::BFloat16 *X_data, const c10::BFloat16 *weight_data, const c10::BFloat16 *bias_data, const c10::BFloat16 *mean_data, const c10::BFloat16 *rstd_data, const int N, const int H, const int W, const int C, const int G, const int64_t act_fn_option, c10::BFloat16 *dx_data, c10::BFloat16 *dweight_data, c10::BFloat16 *dbias_data);
