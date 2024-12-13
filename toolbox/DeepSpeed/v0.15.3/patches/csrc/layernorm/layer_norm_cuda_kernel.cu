#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "type_shim_rope.h"

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()


template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename U> __device__
void cuRMSOnlineSum(
  const U curr,
  U& sigma2)
{
  sigma2 = sigma2 + curr * curr;
}

template<typename U> __device__
void cuChanRMSOnlineSum(
  const U sigma2B,
  U& sigma2)
{
  sigma2 = sigma2 + sigma2B;
}


template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1*n2;
    int l = 4*thrx;
    for (;  l+3 < n2;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      } else {
       cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
#ifndef __ILUVATAR__
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
#else
      U sigma2B = WARP_SHFL(sigma2, srcLaneB, 32);
#endif
      if (!rms_only) {
#ifndef __ILUVATAR__
        U muB = WARP_SHFL(mu, srcLaneB);
        U countB = WARP_SHFL(count, srcLaneB);
#else
        U muB = WARP_SHFL(mu, srcLaneB, 32);
        U countB = WARP_SHFL(count, srcLaneB, 32);
#endif
        cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2*wrt_y+1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            U muB = ubuf[2*threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B,sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
#ifndef __ILUVATAR__
        mu = WARP_SHFL(mu, 0);
#else
        mu = WARP_SHFL(mu, 0, 32);
#endif
      }
#ifndef __ILUVATAR__
      sigma2 = WARP_SHFL(sigma2/U(n2), 0);
#else
      sigma2 = WARP_SHFL(sigma2/U(n2), 0, 32);
#endif
    }
  }
}

template<> __device__
void cuWelfordMuSigma2(
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu= float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const at::Half* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr,mu,sigma2,count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }

      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float2 curr = __half22float2(*((__half2*)(lvals+l+k)));
        if (!rms_only) {
#ifndef __ILUVATAR__
          cuWelfordOnlineSum(curr.x,mu,sigma2,count);
          cuWelfordOnlineSum(curr.y,mu,sigma2,count);
#else
          cuWelfordOnlineSum<float>(curr.x,mu,sigma2,count);
          cuWelfordOnlineSum<float>(curr.y,mu,sigma2,count);
#endif
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (;  l < n2;  ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr,mu,sigma2,count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
#ifndef __ILUVATAR__
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
#else
      float sigma2B = WARP_SHFL(sigma2, srcLaneB, 32);
#endif
      if (!rms_only) {
#ifndef __ILUVATAR__
        float muB = WARP_SHFL(mu, srcLaneB);
        float countB = WARP_SHFL(count, srcLaneB);
#else
        float muB = WARP_SHFL(mu, srcLaneB, 32);
        float countB = WARP_SHFL(count, srcLaneB, 32);
#endif
        cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y+1] = sigma2;
          if (!rms_only) {
            ubuf[2*wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2*threadIdx.y+1];
          if (!rms_only) {
            float muB = ubuf[2*threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1]/float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
#ifndef __ILUVATAR__
        mu = WARP_SHFL(mu, 0);
#else
        mu = WARP_SHFL(mu, 0, 32);
#endif
      }
#ifndef __ILUVATAR__
      sigma2 = WARP_SHFL(sigma2/float(n2), 0);
#else
      sigma2 = WARP_SHFL(sigma2/float(n2), 0, 32);
#endif
    }
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2_opt(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  const int numx = blockDim.x * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const T* lvals = vals + i1*n2;

  #pragma unroll
  for (int l = tid;l < n2;l+=numx){
    U curr = static_cast<U>(lvals[l]);
    if (!rms_only) {
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    } else {
      cuRMSOnlineSum<U>(curr, sigma2);
    }
  }

  U muB;
  U sigma2B;
  U countB;
  #pragma unroll
  for (int offset=32;offset>0;offset/=2) {
    if (rms_only) {
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      sigma2 += sigma2B;
    } else {
      muB = __shfl_xor_sync(0xffffffff, mu, offset, 64);
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      countB = __shfl_xor_sync(0xffffffff, count, offset, 64);
      cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
    }
  }

  if (blockDim.y > 1) {
    if (rms_only) {
      if (tid % 64 == 0) {
        buf[tid/64] = sigma2;
      }
      __syncthreads();
      sigma2 = buf[0];
      for (int i=1;i<numx/64;++i) {
        sigma2 += buf[i];
      }
      sigma2 /= U(n2);
    } else {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (tid % 64 == 0) {
        buf[tid/64*3] = mu;
        buf[tid/64*3+1] = sigma2;
        buf[tid/64*3+2] = count;
      }
      __syncthreads();
      for (;offset>0;offset/=2) {
        if (tid < offset) {
          cuChanOnlineSum<U>(buf[tid*3+offset*3],buf[tid*3+offset*3+1],buf[tid*3+offset*3+2],buf[tid*3],buf[tid*3+1],buf[tid*3+2]);
        }
        __syncthreads();
      }
      mu = buf[0];
      sigma2 = buf[1]/U(n2);
    }
  } else {
    sigma2 /= U(n2);
  }
}


template<> __device__
void cuWelfordMuSigma2_opt(
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  mu = float(0);
  sigma2 = float(0);
  float count = float(0);
  const int numx = blockDim.x * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;

  const float* lvals = reinterpret_cast<const float*>(vals + i1*n2);

  float curr = float(0);
  at::Half* curr1 = reinterpret_cast<at::Half*>(&curr);

  v4u32 aBase;
  aBase.x = (unsigned)(unsigned long long)lvals;
  aBase.y = (unsigned)((unsigned long long)lvals >> 32);
  aBase.zw = -1u;

  #pragma unroll
  for (int l = 0;l < n2/(numx*2);l++){
    curr = __ivcorex_ml_mem_load_f32(aBase, 4 * (tid+l*numx), 0, 0);
    if (rms_only) {
      cuRMSOnlineSum<float>(static_cast<float>(curr1[0]), sigma2);
      cuRMSOnlineSum<float>(static_cast<float>(curr1[1]), sigma2);
    } else {
      cuWelfordOnlineSum<float>(static_cast<float>(curr1[0]), mu, sigma2, count);
      cuWelfordOnlineSum<float>(static_cast<float>(curr1[1]), mu, sigma2, count);
    }
  }

  float muB;
  float sigma2B;
  float countB;
  #pragma unroll
  for (int offset=32;offset>0;offset/=2) {
    if (rms_only) {
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      sigma2 += sigma2B;
    } else {
      muB = __shfl_xor_sync(0xffffffff, mu, offset, 64);
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      countB = __shfl_xor_sync(0xffffffff, count, offset, 64);
      cuChanOnlineSum<float>(muB,sigma2B,countB,mu,sigma2,count);
    }
  }

  if (blockDim.y > 1) {
    if (rms_only) {
      if (tid % 64 == 0) {
        buf[tid/64] = sigma2;
      }
      __syncthreads();
      sigma2 = buf[0];
      for (int i=1;i<numx/64;++i) {
        sigma2 += buf[i];
      }
      sigma2 /= float(n2);
    } else {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (tid % 64 == 0) {
        buf[tid/64*3] = mu;
        buf[tid/64*3+1] = sigma2;
        buf[tid/64*3+2] = count;
      }
      __syncthreads();
      for (;offset>0;offset/=2) {
        if (tid < offset) {
          cuChanOnlineSum<float>(buf[tid*3+offset*3],buf[tid*3+offset*3+1],buf[tid*3+offset*3+2],buf[tid*3],buf[tid*3+1],buf[tid*3+2]);
        }
        __syncthreads();
      }
      mu = buf[0];
      sigma2 = buf[1]/float(n2);
    }
  } else {
    sigma2 /= float(n2);
  }
}

template<typename U> U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template<> float rsqrt(float v) {
  return rsqrtf(v);
}
template<> double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    // cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf,rms_only);
    cuWelfordMuSigma2_opt(vals,n1,n2,i1,mu,sigma2,buf,rms_only);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }

      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2_opt2(
  const T* __restrict__ vals,
  const T* __restrict__ residual,
  const int n1,
  const int n2,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf,
  bool rms_only)
{
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  const int numx = blockDim.x * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const T* lvals = vals + i1*n2;
  const T* lresidual = residual + i1*n2;

  #pragma unroll
  for (int l = tid;l < n2;l+=numx){
    U curr = static_cast<U>(lvals[l]+lresidual[l]);
    if (!rms_only) {
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    } else {
      cuRMSOnlineSum<U>(curr, sigma2);
    }
  }

  U muB;
  U sigma2B;
  U countB;
  #pragma unroll
  for (int offset=32;offset>0;offset/=2) {
    if (rms_only) {
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      sigma2 += sigma2B;
    } else {
      muB = __shfl_xor_sync(0xffffffff, mu, offset, 64);
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      countB = __shfl_xor_sync(0xffffffff, count, offset, 64);
      cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
    }
  }

  if (blockDim.y > 1) {
    if (rms_only) {
      if (tid % 64 == 0) {
        buf[tid/64] = sigma2;
      }
      __syncthreads();
      sigma2 = buf[0];
      for (int i=1;i<numx/64;++i) {
        sigma2 += buf[i];
      }
      sigma2 /= U(n2);
    } else {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (tid % 64 == 0) {
        buf[tid/64*3] = mu;
        buf[tid/64*3+1] = sigma2;
        buf[tid/64*3+2] = count;
      }
      __syncthreads();
      for (;offset>0;offset/=2) {
        if (tid < offset) {
          cuChanOnlineSum<U>(buf[tid*3+offset*3],buf[tid*3+offset*3+1],buf[tid*3+offset*3+2],buf[tid*3],buf[tid*3+1],buf[tid*3+2]);
        }
        __syncthreads();
      }
      mu = buf[0];
      sigma2 = buf[1]/U(n2);
    }
  } else {
    sigma2 /= U(n2);
  }
}


template<> __device__
void cuWelfordMuSigma2_opt2(
  const at::Half* __restrict__ vals,
  const at::Half* __restrict__ residual,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  mu = float(0);
  sigma2 = float(0);
  float count = float(0);
  const int numx = blockDim.x * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;

  const float* lvals = reinterpret_cast<const float*>(vals + i1*n2);
  const float* lresidual = reinterpret_cast<const float*>(residual + i1*n2);

  float curr1 = float(0);
  at::Half* curr1_ = reinterpret_cast<at::Half*>(&curr1);
  float curr2 = float(0);
  at::Half* curr2_ = reinterpret_cast<at::Half*>(&curr2);

  v4u32 aBase;
  aBase.x = (unsigned)(unsigned long long)lvals;
  aBase.y = (unsigned)((unsigned long long)lvals >> 32);
  aBase.zw = -1u;

  v4u32 bBase;
  bBase.x = (unsigned)(unsigned long long)lresidual;
  bBase.y = (unsigned)((unsigned long long)lresidual >> 32);
  bBase.zw = -1u;

  #pragma unroll
  for (int l = 0;l < n2/(numx*2);l++){
    curr1 = __ivcorex_ml_mem_load_f32(aBase, 4 * (tid+l*numx), 0, 0);
    curr2 = __ivcorex_ml_mem_load_f32(bBase, 4 * (tid+l*numx), 0, 0);
    curr1_[0] += curr2_[0];
    curr1_[1] += curr2_[1];

    if (rms_only) {
      cuRMSOnlineSum<float>(static_cast<float>(curr1_[0]), sigma2);
      cuRMSOnlineSum<float>(static_cast<float>(curr1_[1]), sigma2);
    } else {
      cuWelfordOnlineSum<float>(static_cast<float>(curr1_[0]), mu, sigma2, count);
      cuWelfordOnlineSum<float>(static_cast<float>(curr1_[1]), mu, sigma2, count);
    }
  }

  float muB;
  float sigma2B;
  float countB;
  #pragma unroll
  for (int offset=32;offset>0;offset/=2) {
    if (rms_only) {
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      sigma2 += sigma2B;
    } else {
      muB = __shfl_xor_sync(0xffffffff, mu, offset, 64);
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      countB = __shfl_xor_sync(0xffffffff, count, offset, 64);
      cuChanOnlineSum<float>(muB,sigma2B,countB,mu,sigma2,count);
    }
  }

  if (blockDim.y > 1) {
    if (rms_only) {
      if (tid % 64 == 0) {
        buf[tid/64] = sigma2;
      }
      __syncthreads();
      sigma2 = buf[0];
      for (int i=1;i<numx/64;++i) {
        sigma2 += buf[i];
      }
      sigma2 /= float(n2);
    } else {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (tid % 64 == 0) {
        buf[tid/64*3] = mu;
        buf[tid/64*3+1] = sigma2;
        buf[tid/64*3+2] = count;
      }
      __syncthreads();
      for (;offset>0;offset/=2) {
        if (tid < offset) {
          cuChanOnlineSum<float>(buf[tid*3+offset*3],buf[tid*3+offset*3+1],buf[tid*3+offset*3+2],buf[tid*3],buf[tid*3+1],buf[tid*3+2]);
        }
        __syncthreads();
      }
      mu = buf[0];
      sigma2 = buf[1]/float(n2);
    }
  } else {
    sigma2 /= float(n2);
  }
}

template<> __device__
void cuWelfordMuSigma2_opt2(
  const at::BFloat16* __restrict__ vals,
  const at::BFloat16* __restrict__ residual,
  const int n1,
  const int n2,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf,
  bool rms_only)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  mu = float(0);
  sigma2 = float(0);
  float count = float(0);
  const int numx = blockDim.x * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;

  const float* lvals = reinterpret_cast<const float*>(vals + i1*n2);
  const float* lresidual = reinterpret_cast<const float*>(residual + i1*n2);

  float curr1 = float(0);
  at::BFloat16* curr1_ = reinterpret_cast<at::BFloat16*>(&curr1);
  float curr2 = float(0);
  at::BFloat16* curr2_ = reinterpret_cast<at::BFloat16*>(&curr2);

  v4u32 aBase;
  aBase.x = (unsigned)(unsigned long long)lvals;
  aBase.y = (unsigned)((unsigned long long)lvals >> 32);
  aBase.zw = -1u;

  v4u32 bBase;
  bBase.x = (unsigned)(unsigned long long)lresidual;
  bBase.y = (unsigned)((unsigned long long)lresidual >> 32);
  bBase.zw = -1u;

  #pragma unroll
  for (int l = 0;l < n2/(numx*2);l++){
    curr1 = __ivcorex_ml_mem_load_f32(aBase, 4 * (tid+l*numx), 0, 0);
    curr2 = __ivcorex_ml_mem_load_f32(bBase, 4 * (tid+l*numx), 0, 0);
    curr1_[0] += curr2_[0];
    curr1_[1] += curr2_[1];

    if (rms_only) {
      cuRMSOnlineSum<float>(static_cast<float>(curr1_[0]), sigma2);
      cuRMSOnlineSum<float>(static_cast<float>(curr1_[1]), sigma2);
    } else {
      cuWelfordOnlineSum<float>(static_cast<float>(curr1_[0]), mu, sigma2, count);
      cuWelfordOnlineSum<float>(static_cast<float>(curr1_[1]), mu, sigma2, count);
    }
  }

  float muB;
  float sigma2B;
  float countB;
  #pragma unroll
  for (int offset=32;offset>0;offset/=2) {
    if (rms_only) {
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      sigma2 += sigma2B;
    } else {
      muB = __shfl_xor_sync(0xffffffff, mu, offset, 64);
      sigma2B = __shfl_xor_sync(0xffffffff, sigma2, offset, 64);
      countB = __shfl_xor_sync(0xffffffff, count, offset, 64);
      cuChanOnlineSum<float>(muB,sigma2B,countB,mu,sigma2,count);
    }
  }

  if (blockDim.y > 1) {
    if (rms_only) {
      if (tid % 64 == 0) {
        buf[tid/64] = sigma2;
      }
      __syncthreads();
      sigma2 = buf[0];
      for (int i=1;i<numx/64;++i) {
        sigma2 += buf[i];
      }
      sigma2 /= float(n2);
    } else {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (tid % 64 == 0) {
        buf[tid/64*3] = mu;
        buf[tid/64*3+1] = sigma2;
        buf[tid/64*3+2] = count;
      }
      __syncthreads();
      for (;offset>0;offset/=2) {
        if (tid < offset) {
          cuChanOnlineSum<float>(buf[tid*3+offset*3],buf[tid*3+offset*3+1],buf[tid*3+offset*3+2],buf[tid*3],buf[tid*3+1],buf[tid*3+2]);
        }
        __syncthreads();
      }
      mu = buf[0];
      sigma2 = buf[1]/float(n2);
    }
  } else {
    sigma2 /= float(n2);
  }
}

template<typename T, typename U, typename V> __device__
void cuApplyLayerNormRes_(
  V* __restrict__ output_vals,
  V* __restrict__ output_sum,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const T* __restrict__ residual,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2_opt2(vals,residual,n1,n2,i1,mu,sigma2,buf,rms_only);

    __syncthreads();
    const T* lvals = vals + i1*n2;
    const T* lresidual = residual + i1*n2;
    V* ovals = output_vals + i1*n2;
    V* osum = output_sum + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]+lresidual[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }
        osum[i] = static_cast<T>(curr);
      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]+lresidual[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
        osum[i] = static_cast<T>(curr);
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename T, typename U, typename V=T> __global__
void cuApplyLayerNorm(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta
  )
{
  cuApplyLayerNorm_<T, U, V>(output_vals, mean, invvar, vals, n1, n2, epsilon, gamma, beta, false);
}

template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNorm(
  V* __restrict__ output_vals,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma)
{
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, invvar, vals, n1, n2, epsilon, gamma, NULL, true);
}

template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNormRes(
  V* __restrict__ output_vals,
  V* __restrict__ output_sum,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const T* __restrict__ residual,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma)
{
  cuApplyLayerNormRes_<T, U, V>(output_vals, output_sum, NULL, invvar, vals, residual, n1, n2, epsilon, gamma, NULL, true);
}

template<typename T, typename U, typename V> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] = curr_dout;
          warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] = curr_dout * (curr_input) * curr_invvar;
        }
      } else {
        if (!rms_only) {
          warp_buf1[write_idx] = U(0);
        }
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (!rms_only) {
        warp_buf1[write_idx] = U(0);
      }
      warp_buf2[write_idx] = U(0);
    }
  }
}

template<typename T, typename U, typename V> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    bool rms_only
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] += curr_dout;
          warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] += curr_dout * (curr_input) * curr_invvar;
        }
      }
    }
  }
}


template<typename T, typename U, typename V> __global__
void cuComputePartGradGammaBeta(
    const V* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    U* part_grad_gamma,
    U* part_grad_beta,
    bool rms_only)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,n2,mean,invvar, rms_only);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,n2,mean,invvar, rms_only);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      if (!rms_only) {
        acc1 += warp_buf1[idx1];
      }
      acc2 += warp_buf2[idx1];
    }
    if (!rms_only) {
      warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    }
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        if (!rms_only) {
          warp_buf1[idx1] += warp_buf1[idx2];
        }
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      if (!rms_only) {
        part_grad_beta[blockIdx.y*n2+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      }
      part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename U, typename V> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma,
    V* grad_beta,
    bool rms_only)
{
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_gamma = U(0);
      U sum_beta = U(0);
      const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
      const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*n2];
        if (!rms_only) {
          sum_beta += part_grad_beta_ptr[warp_offset*n2];
        }
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;
          if (!rms_only) {
            buf[write_idx+nbsize3] = sum_beta;
          }
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];
          if (!rms_only) {
            sum_beta += buf[read_idx+nbsize3];
          }
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        if (!rms_only) {
          grad_beta[i2] = sum_beta;
        }
      }
    }
}


template<typename T, typename U, typename V> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    T* grad_input,
    bool rms_only)
{
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    U c_mean;
    if (!rms_only) {
      c_mean = mean[i1];
    }
    const U c_invvar = invvar[i1];
    const T* k_input = input + i1*n2;
    const V* k_dout = dout + i1*n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss * gamma[l+k];
            sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * gamma[l+k] * (c_h) * c_invvar;
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * gamma[l];
          sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * gamma[l] * (c_h) * c_invvar;
        }

      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * (c_h) * c_invvar;
          }
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * (c_h) * c_invvar;
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      if (!rms_only) {
#ifndef __ILUVATAR__
        sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
#else
        sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask, 32);
#endif
      }
#ifndef __ILUVATAR__
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
#else
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask, 32);
#endif
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          if (!rms_only) {
            buf[2*wrt_i] = sum_loss1;
          }
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          if (!rms_only) {
            sum_loss1 += buf[2*read_i];
          }
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        if (!rms_only) {
          buf[2*threadIdx.x] = sum_loss1;
        }
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        if (!rms_only) {
          sum_loss1 = buf[2*threadIdx.x];
        }
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * gamma[l];
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h) * c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h) * c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

template<typename V> __device__
V clamp_by_magnitude(V curr_gamma, double eps)
{
  const V kMinGamma = V(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}

template<int LDG, bool RMSONLY, bool MemoryEfficient, typename T, typename U, typename V> __global__
void fusedGradInputWeights(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* gamma,
    const V* beta,
    const double eps,
    T* grad_input,
    U* part_grad_gamma,
    U* part_grad_beta)
{
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  U d_gamma[LDG] = {0};
  U d_beta[LDG] = {0};
  __shared__ U shm[80];

  V gamma_data[LDG];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    gamma_data[l] = gamma[thrx+l*numx];
  }

  V beta_data[LDG];
  if (MemoryEfficient && !RMSONLY) {
    #pragma unroll
    for (int l=0;l<LDG;l++) {
      beta_data[l] = beta[thrx+l*numx];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_invvar = invvar[row];
    const T* input_or_output_ptr = input_or_output + row*n2;
    const V* dout_ptr = dout + row*n2;
    
    U y[LDG];
    U dy[LDG];

    // data reuse
    for (int l=0;l<LDG;l++) {
      const U c_h = static_cast<U>(input_or_output_ptr[thrx+l*numx]);
      const U c_loss = static_cast<U>(dout_ptr[thrx+l*numx]);
      U y_tmp;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp = (c_h - mean[row]) * c_invvar;
        } else {
          y_tmp = (c_h - static_cast<U>(beta_data[l])) / static_cast<U>(clamp_by_magnitude(gamma_data[l], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp = c_h * c_invvar;
        } else {
          y_tmp = c_h / static_cast<U>(clamp_by_magnitude(gamma_data[l], eps));
        }
      }
      U dy_tmp = c_loss * gamma_data[l];
      if (!RMSONLY) {
        sum_loss1 += dy_tmp;
      }
      sum_loss2 += dy_tmp * y_tmp;

      y[l] = y_tmp;
      dy[l] = dy_tmp;
      d_gamma[l] += c_loss * y_tmp;
      if (!RMSONLY) {
        d_beta[l] += c_loss;
      }
    }

    // intra warp reduction
    U val1;
    U val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + row*n2;
    for (int l=0;l<LDG;l++) {
      U y_tmp = y[l];
      U dy_tmp = dy[l];
      U f_grad_input = fH * dy_tmp;
      if (!RMSONLY) {
        f_grad_input -= sum_loss1 + y_tmp * sum_loss2;
      } else {
        f_grad_input -= y_tmp * sum_loss2;
      }
      f_grad_input *= term1;
      k_grad_input[thrx+l*numx] = static_cast<T>(f_grad_input);
    }
  }
  // #pragma unroll
  for (int l=0;l<LDG;l++) {
    part_grad_gamma[blockIdx.y*n2+l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta[blockIdx.y*n2+l*numx+thrx] = d_beta[l];
    }
  }
}



template<int LDG, bool RMSONLY, bool MemoryEfficient> __global__
void fusedGradInputWeights(
    const at::Half* __restrict__ dout,
    const at::Half* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    const at::Half* gamma,
    const at::Half* beta,
    const double eps,
    at::Half* grad_input,
    float* part_grad_gamma,
    float* part_grad_beta)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  float2 d_gamma[LDG] = {0};
  float2 d_beta[LDG] = {0};
  
  __shared__ float shm[80];

  float c_h = float(0);
  at::Half* c_h1 = reinterpret_cast<at::Half*>(&c_h);
  float c_loss = float(0);
  at::Half* c_loss1 = reinterpret_cast<at::Half*>(&c_loss);
  float c_gamma = float(0);
  at::Half* c_gamma1 = reinterpret_cast<at::Half*>(&c_gamma);

  v4u32 gBase;
  gBase.x = (unsigned)(unsigned long long)gamma;
  gBase.y = (unsigned)((unsigned long long)gamma >> 32);
  gBase.zw = -1u;

  at::Half gamma_data[LDG*2];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    c_gamma = __ivcorex_ml_mem_load_f32(gBase, 4*(thrx+l*numx), 0, 0);
    gamma_data[l*2] = c_gamma1[0];
    gamma_data[l*2+1] = c_gamma1[1];
  }

  at::Half beta_data[LDG*2];
  if (MemoryEfficient && !RMSONLY) {
    float c_beta = float(0);
    at::Half* c_beta1 = reinterpret_cast<at::Half*>(&c_beta);

    v4u32 hBase;
    hBase.x = (unsigned)(unsigned long long)beta;
    hBase.y = (unsigned)((unsigned long long)beta >> 32);
    hBase.zw = -1u;

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_beta = __ivcorex_ml_mem_load_f32(hBase, 4*(thrx+l*numx), 0, 0);
      beta_data[l*2] = c_beta1[0];
      beta_data[l*2+1] = c_beta1[1];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);

    const float c_invvar = invvar[row];
    const float* input_or_output_ptr = reinterpret_cast<const float*>(input_or_output + row*n2);
    const float* dout_ptr = reinterpret_cast<const float*>(dout + row*n2);

    v4u32 aBase;
    aBase.x = (unsigned)(unsigned long long)input_or_output_ptr;
    aBase.y = (unsigned)((unsigned long long)input_or_output_ptr >> 32);
    aBase.zw = -1u;

    v4u32 bBase;
    bBase.x = (unsigned)(unsigned long long)dout_ptr;
    bBase.y = (unsigned)((unsigned long long)dout_ptr >> 32);
    bBase.zw = -1u;
    
    float2 y[LDG];
    float2 dy[LDG];

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_h = __ivcorex_ml_mem_load_f32(aBase, 4 * (thrx+l*numx), 0, 0);
      c_loss = __ivcorex_ml_mem_load_f32(bBase, 4 * (thrx+l*numx), 0, 0);
      float y_tmp0;
      float y_tmp1;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp0 = (static_cast<float>(c_h1[0]) - mean[row]) * c_invvar;
          y_tmp1 = (static_cast<float>(c_h1[1]) - mean[row]) * c_invvar;
        } else {
          y_tmp0 = (static_cast<float>(c_h1[0]) - static_cast<float>(beta_data[l*2])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = (static_cast<float>(c_h1[1]) - static_cast<float>(beta_data[l*2+1])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp0 = static_cast<float>(c_h1[0]) * c_invvar;
          y_tmp1 = static_cast<float>(c_h1[1]) * c_invvar;
        } else {
          y_tmp0 = static_cast<float>(c_h1[0]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = static_cast<float>(c_h1[1]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      }
      float dy_tmp0 = static_cast<float>(c_loss1[0]) * static_cast<float>(gamma_data[l*2]);
      float dy_tmp1 = static_cast<float>(c_loss1[1]) * static_cast<float>(gamma_data[l*2+1]);
      sum_loss1 += dy_tmp0 + dy_tmp1;
      sum_loss2 += y_tmp0 * dy_tmp0 + y_tmp1 * dy_tmp1;
      y[l].x = y_tmp0;
      y[l].y = y_tmp1;
      dy[l].x = dy_tmp0;
      dy[l].y = dy_tmp1;
      d_gamma[l].x += static_cast<float>(c_loss1[0]) * y_tmp0;
      d_gamma[l].y += static_cast<float>(c_loss1[1]) * y_tmp1;
      if (!RMSONLY) {
        d_beta[l].x += static_cast<float>(c_loss1[0]);
        d_beta[l].y += static_cast<float>(c_loss1[1]);
      }
    }

    // intra warp reduction
    float val1;
    float val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float* k_grad_input = reinterpret_cast<float*>(grad_input + row*n2);
    for (int l=0;l<LDG;l++) {
      float2 y_tmp;
      y_tmp.x = y[l].x;
      y_tmp.y = y[l].y;
      float2 dy_tmp;
      dy_tmp.x = dy[l].x;
      dy_tmp.y = dy[l].y;

      float f_grad_input0 = fH * dy_tmp.x;
      float f_grad_input1 = fH * dy_tmp.y;
      if (!RMSONLY) {
        f_grad_input0 -= sum_loss1 + y_tmp.x * sum_loss2;
        f_grad_input1 -= sum_loss1 + y_tmp.y * sum_loss2;
      } else {
        f_grad_input0 -= y_tmp.x * sum_loss2;
        f_grad_input1 -= y_tmp.y * sum_loss2;
      }
      f_grad_input0 *= term1;
      f_grad_input1 *= term1;

      float f_grad_input;
      at::Half* f_grad_input_ = reinterpret_cast<at::Half*>(&f_grad_input);
      f_grad_input_[0] = static_cast<at::Half>(f_grad_input0);
      f_grad_input_[1] = static_cast<at::Half>(f_grad_input1);
      k_grad_input[thrx+l*numx] = f_grad_input;
    }
  }
  
  float2* part_grad_gamma_ptr = reinterpret_cast<float2*>(part_grad_gamma + blockIdx.y*n2);
  float2* part_grad_beta_ptr = reinterpret_cast<float2*>(part_grad_beta + blockIdx.y*n2);
  for (int l=0;l<LDG;l++) {
    part_grad_gamma_ptr[l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta_ptr[l*numx+thrx] = d_beta[l];
    }
  }
}

template<int LDG, bool RMSONLY, bool MemoryEfficient> __global__
void fusedGradInputWeights(
    const at::BFloat16* __restrict__ dout,
    const at::BFloat16* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    const at::BFloat16* gamma,
    const at::BFloat16* beta,
    const double eps,
    at::BFloat16* grad_input,
    float* part_grad_gamma,
    float* part_grad_beta)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  float2 d_gamma[LDG] = {0};
  float2 d_beta[LDG] = {0};
  
  __shared__ float shm[80];

  float c_h = float(0);
  at::BFloat16* c_h1 = reinterpret_cast<at::BFloat16*>(&c_h);
  float c_loss = float(0);
  at::BFloat16* c_loss1 = reinterpret_cast<at::BFloat16*>(&c_loss);
  float c_gamma = float(0);
  at::BFloat16* c_gamma1 = reinterpret_cast<at::BFloat16*>(&c_gamma);

  v4u32 gBase;
  gBase.x = (unsigned)(unsigned long long)gamma;
  gBase.y = (unsigned)((unsigned long long)gamma >> 32);
  gBase.zw = -1u;

  at::BFloat16 gamma_data[LDG*2];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    c_gamma = __ivcorex_ml_mem_load_f32(gBase, 4*(thrx+l*numx), 0, 0);
    gamma_data[l*2] = c_gamma1[0];
    gamma_data[l*2+1] = c_gamma1[1];
  }

  at::BFloat16 beta_data[LDG*2];
  if (MemoryEfficient && !RMSONLY) {
    float c_beta = float(0);
    at::BFloat16* c_beta1 = reinterpret_cast<at::BFloat16*>(&c_beta);

    v4u32 hBase;
    hBase.x = (unsigned)(unsigned long long)beta;
    hBase.y = (unsigned)((unsigned long long)beta >> 32);
    hBase.zw = -1u;

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_beta = __ivcorex_ml_mem_load_f32(hBase, 4*(thrx+l*numx), 0, 0);
      beta_data[l*2] = c_beta1[0];
      beta_data[l*2+1] = c_beta1[1];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);

    const float c_invvar = invvar[row];
    const float* input_or_output_ptr = reinterpret_cast<const float*>(input_or_output + row*n2);
    const float* dout_ptr = reinterpret_cast<const float*>(dout + row*n2);

    v4u32 aBase;
    aBase.x = (unsigned)(unsigned long long)input_or_output_ptr;
    aBase.y = (unsigned)((unsigned long long)input_or_output_ptr >> 32);
    aBase.zw = -1u;

    v4u32 bBase;
    bBase.x = (unsigned)(unsigned long long)dout_ptr;
    bBase.y = (unsigned)((unsigned long long)dout_ptr >> 32);
    bBase.zw = -1u;
    
    float2 y[LDG];
    float2 dy[LDG];

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_h = __ivcorex_ml_mem_load_f32(aBase, 4 * (thrx+l*numx), 0, 0);
      c_loss = __ivcorex_ml_mem_load_f32(bBase, 4 * (thrx+l*numx), 0, 0);
      float y_tmp0;
      float y_tmp1;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp0 = (static_cast<float>(c_h1[0]) - mean[row]) * c_invvar;
          y_tmp1 = (static_cast<float>(c_h1[1]) - mean[row]) * c_invvar;
        } else {
          y_tmp0 = (static_cast<float>(c_h1[0]) - static_cast<float>(beta_data[l*2])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = (static_cast<float>(c_h1[1]) - static_cast<float>(beta_data[l*2+1])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp0 = static_cast<float>(c_h1[0]) * c_invvar;
          y_tmp1 = static_cast<float>(c_h1[1]) * c_invvar;
        } else {
          y_tmp0 = static_cast<float>(c_h1[0]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = static_cast<float>(c_h1[1]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      }
      float dy_tmp0 = static_cast<float>(c_loss1[0]) * static_cast<float>(gamma_data[l*2]);
      float dy_tmp1 = static_cast<float>(c_loss1[1]) * static_cast<float>(gamma_data[l*2+1]);
      sum_loss1 += dy_tmp0 + dy_tmp1;
      sum_loss2 += y_tmp0 * dy_tmp0 + y_tmp1 * dy_tmp1;
      y[l].x = y_tmp0;
      y[l].y = y_tmp1;
      dy[l].x = dy_tmp0;
      dy[l].y = dy_tmp1;
      d_gamma[l].x += static_cast<float>(c_loss1[0]) * y_tmp0;
      d_gamma[l].y += static_cast<float>(c_loss1[1]) * y_tmp1;
      if (!RMSONLY) {
        d_beta[l].x += static_cast<float>(c_loss1[0]);
        d_beta[l].y += static_cast<float>(c_loss1[1]);
      }
    }

    // intra warp reduction
    float val1;
    float val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float* k_grad_input = reinterpret_cast<float*>(grad_input + row*n2);
    for (int l=0;l<LDG;l++) {
      float2 y_tmp;
      y_tmp.x = y[l].x;
      y_tmp.y = y[l].y;
      float2 dy_tmp;
      dy_tmp.x = dy[l].x;
      dy_tmp.y = dy[l].y;

      float f_grad_input0 = fH * dy_tmp.x;
      float f_grad_input1 = fH * dy_tmp.y;
      if (!RMSONLY) {
        f_grad_input0 -= sum_loss1 + y_tmp.x * sum_loss2;
        f_grad_input1 -= sum_loss1 + y_tmp.y * sum_loss2;
      } else {
        f_grad_input0 -= y_tmp.x * sum_loss2;
        f_grad_input1 -= y_tmp.y * sum_loss2;
      }
      f_grad_input0 *= term1;
      f_grad_input1 *= term1;

      float f_grad_input;
      at::BFloat16* f_grad_input_ = reinterpret_cast<at::BFloat16*>(&f_grad_input);
      f_grad_input_[0] = static_cast<at::BFloat16>(f_grad_input0);
      f_grad_input_[1] = static_cast<at::BFloat16>(f_grad_input1);
      k_grad_input[thrx+l*numx] = f_grad_input;
    }
  }
  
  float2* part_grad_gamma_ptr = reinterpret_cast<float2*>(part_grad_gamma + blockIdx.y*n2);
  float2* part_grad_beta_ptr = reinterpret_cast<float2*>(part_grad_beta + blockIdx.y*n2);
  for (int l=0;l<LDG;l++) {
    part_grad_gamma_ptr[l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta_ptr[l*numx+thrx] = d_beta[l];
    }
  }
}

template<int LDG, bool RMSONLY, bool MemoryEfficient, typename T, typename U, typename V> __global__
void fusedGradInputWeights_(
    const V* __restrict__ dout,
    const V* __restrict__ dres,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    const V* gamma,
    const V* beta,
    const double eps,
    T* grad_input,
    U* part_grad_gamma,
    U* part_grad_beta)
{
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  U d_gamma[LDG] = {0};
  U d_beta[LDG] = {0};
  __shared__ U shm[80];

  V gamma_data[LDG];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    gamma_data[l] = gamma[thrx+l*numx];
  }

  V beta_data[LDG];
  if (MemoryEfficient && !RMSONLY) {
    #pragma unroll
    for (int l=0;l<LDG;l++) {
      beta_data[l] = beta[thrx+l*numx];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_invvar = invvar[row];
    const T* input_or_output_ptr = input_or_output + row*n2;
    const V* dout_ptr = dout + row*n2;
    const V* dres_ptr = dres + row*n2;
    
    U y[LDG];
    U dy[LDG];

    // data reuse
    for (int l=0;l<LDG;l++) {
      const U c_h = static_cast<U>(input_or_output_ptr[thrx+l*numx]);
      const U c_loss = static_cast<U>(dout_ptr[thrx+l*numx]);
      U y_tmp;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp = (c_h - mean[row]) * c_invvar;
        } else {
          y_tmp = (c_h - static_cast<U>(beta_data[l])) / static_cast<U>(clamp_by_magnitude(gamma_data[l], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp = c_h * c_invvar;
        } else {
          y_tmp = c_h / static_cast<U>(clamp_by_magnitude(gamma_data[l], eps));
        }
      }
      U dy_tmp = c_loss * gamma_data[l];
      if (!RMSONLY) {
        sum_loss1 += dy_tmp;
      }
      sum_loss2 += dy_tmp * y_tmp;

      y[l] = y_tmp;
      dy[l] = dy_tmp;
      d_gamma[l] += c_loss * y_tmp;
      if (!RMSONLY) {
        d_beta[l] += c_loss;
      }
    }

    // intra warp reduction
    U val1;
    U val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + row*n2;
    for (int l=0;l<LDG;l++) {
      U y_tmp = y[l];
      U dy_tmp = dy[l];
      U f_grad_input = fH * dy_tmp;
      if (!RMSONLY) {
        f_grad_input -= sum_loss1 + y_tmp * sum_loss2;
      } else {
        f_grad_input -= y_tmp * sum_loss2;
      }
      f_grad_input *= term1;
      f_grad_input += static_cast<U>(dres_ptr[thrx+l*numx]);
      k_grad_input[thrx+l*numx] = static_cast<T>(f_grad_input);
    }
  }
  // #pragma unroll
  for (int l=0;l<LDG;l++) {
    part_grad_gamma[blockIdx.y*n2+l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta[blockIdx.y*n2+l*numx+thrx] = d_beta[l];
    }
  }
}

template<int LDG, bool RMSONLY, bool MemoryEfficient> __global__
void fusedGradInputWeights_(
    const at::Half* __restrict__ dout,
    const at::Half* __restrict__ dres,
    const at::Half* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    const at::Half* gamma,
    const at::Half* beta,
    const double eps,
    at::Half* grad_input,
    float* part_grad_gamma,
    float* part_grad_beta)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  float2 d_gamma[LDG] = {0};
  float2 d_beta[LDG] = {0};
  
  __shared__ float shm[80];

  float c_h = float(0);
  at::Half* c_h1 = reinterpret_cast<at::Half*>(&c_h);
  float c_loss = float(0);
  at::Half* c_loss1 = reinterpret_cast<at::Half*>(&c_loss);
  float c_dres = float(0);
  at::Half* c_dres1 = reinterpret_cast<at::Half*>(&c_dres);
  float c_gamma = float(0);
  at::Half* c_gamma1 = reinterpret_cast<at::Half*>(&c_gamma);

  v4u32 gBase;
  gBase.x = (unsigned)(unsigned long long)gamma;
  gBase.y = (unsigned)((unsigned long long)gamma >> 32);
  gBase.zw = -1u;

  at::Half gamma_data[LDG*2];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    c_gamma = __ivcorex_ml_mem_load_f32(gBase, 4*(thrx+l*numx), 0, 0);
    gamma_data[l*2] = c_gamma1[0];
    gamma_data[l*2+1] = c_gamma1[1];
  }

  at::Half beta_data[LDG*2];
  if (MemoryEfficient && !RMSONLY) {
    float c_beta = float(0);
    at::Half* c_beta1 = reinterpret_cast<at::Half*>(&c_beta);

    v4u32 hBase;
    hBase.x = (unsigned)(unsigned long long)beta;
    hBase.y = (unsigned)((unsigned long long)beta >> 32);
    hBase.zw = -1u;

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_beta = __ivcorex_ml_mem_load_f32(hBase, 4*(thrx+l*numx), 0, 0);
      beta_data[l*2] = c_beta1[0];
      beta_data[l*2+1] = c_beta1[1];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);
    float c_mean;
    if (!RMSONLY) {
      c_mean = mean[row];
    }
    const float c_invvar = invvar[row];
    const float* input_or_output_ptr = reinterpret_cast<const float*>(input_or_output + row*n2);
    const float* dout_ptr = reinterpret_cast<const float*>(dout + row*n2);
    const float* dres_ptr = reinterpret_cast<const float*>(dres + row*n2);

    v4u32 aBase;
    aBase.x = (unsigned)(unsigned long long)input_or_output_ptr;
    aBase.y = (unsigned)((unsigned long long)input_or_output_ptr >> 32);
    aBase.zw = -1u;

    v4u32 bBase;
    bBase.x = (unsigned)(unsigned long long)dout_ptr;
    bBase.y = (unsigned)((unsigned long long)dout_ptr >> 32);
    bBase.zw = -1u;

    v4u32 cBase;
    cBase.x = (unsigned)(unsigned long long)dres_ptr;
    cBase.y = (unsigned)((unsigned long long)dres_ptr >> 32);
    cBase.zw = -1u;
    
    float2 y[LDG];
    float2 dy[LDG];

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_h = __ivcorex_ml_mem_load_f32(aBase, 4 * (thrx+l*numx), 0, 0);
      c_loss = __ivcorex_ml_mem_load_f32(bBase, 4 * (thrx+l*numx), 0, 0);
      float y_tmp0;
      float y_tmp1;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp0 = (static_cast<float>(c_h1[0]) - mean[row]) * c_invvar;
          y_tmp1 = (static_cast<float>(c_h1[1]) - mean[row]) * c_invvar;
        } else {
          y_tmp0 = (static_cast<float>(c_h1[0]) - static_cast<float>(beta_data[l*2])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = (static_cast<float>(c_h1[1]) - static_cast<float>(beta_data[l*2+1])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp0 = static_cast<float>(c_h1[0]) * c_invvar;
          y_tmp1 = static_cast<float>(c_h1[1]) * c_invvar;
        } else {
          y_tmp0 = static_cast<float>(c_h1[0]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = static_cast<float>(c_h1[1]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      }
      float dy_tmp0 = static_cast<float>(c_loss1[0]) * static_cast<float>(gamma_data[l*2]);
      float dy_tmp1 = static_cast<float>(c_loss1[1]) * static_cast<float>(gamma_data[l*2+1]);
      sum_loss1 += dy_tmp0 + dy_tmp1;
      sum_loss2 += y_tmp0 * dy_tmp0 + y_tmp1 * dy_tmp1;
      y[l].x = y_tmp0;
      y[l].y = y_tmp1;
      dy[l].x = dy_tmp0;
      dy[l].y = dy_tmp1;
      d_gamma[l].x += static_cast<float>(c_loss1[0]) * y_tmp0;
      d_gamma[l].y += static_cast<float>(c_loss1[1]) * y_tmp1;
      if (!RMSONLY) {
        d_beta[l].x += static_cast<float>(c_loss1[0]);
        d_beta[l].y += static_cast<float>(c_loss1[1]);
      }
    }

    // intra warp reduction
    float val1;
    float val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float* k_grad_input = reinterpret_cast<float*>(grad_input + row*n2);
    for (int l=0;l<LDG;l++) {
      float2 y_tmp;
      y_tmp.x = y[l].x;
      y_tmp.y = y[l].y;
      float2 dy_tmp;
      dy_tmp.x = dy[l].x;
      dy_tmp.y = dy[l].y;

      float f_grad_input0 = fH * dy_tmp.x;
      float f_grad_input1 = fH * dy_tmp.y;
      if (!RMSONLY) {
        f_grad_input0 -= sum_loss1 + y_tmp.x * sum_loss2;
        f_grad_input1 -= sum_loss1 + y_tmp.y * sum_loss2;
      } else {
        f_grad_input0 -= y_tmp.x * sum_loss2;
        f_grad_input1 -= y_tmp.y * sum_loss2;
      }
      f_grad_input0 *= term1;
      f_grad_input1 *= term1;

      c_dres = __ivcorex_ml_mem_load_f32(cBase, 4 * (thrx+l*numx), 0, 0);
      float f_grad_input;
      at::Half* f_grad_input_ = reinterpret_cast<at::Half*>(&f_grad_input);
      f_grad_input_[0] = static_cast<at::Half>(f_grad_input0+static_cast<float>(c_dres1[0]));
      f_grad_input_[1] = static_cast<at::Half>(f_grad_input1+static_cast<float>(c_dres1[1]));
      k_grad_input[thrx+l*numx] = f_grad_input;
    }
  }
  
  float2* part_grad_gamma_ptr = reinterpret_cast<float2*>(part_grad_gamma + blockIdx.y*n2);
  float2* part_grad_beta_ptr = reinterpret_cast<float2*>(part_grad_beta + blockIdx.y*n2);
  for (int l=0;l<LDG;l++) {
    part_grad_gamma_ptr[l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta_ptr[l*numx+thrx] = d_beta[l];
    }
  }
}

template<int LDG, bool RMSONLY, bool MemoryEfficient> __global__
void fusedGradInputWeights_(
    const at::BFloat16* __restrict__ dout,
    const at::BFloat16* __restrict__ dres,
    const at::BFloat16* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    const at::BFloat16* gamma,
    const at::BFloat16* beta,
    const double eps,
    at::BFloat16* grad_input,
    float* part_grad_gamma,
    float* part_grad_beta)
{
  typedef unsigned v4u32 __attribute__((ext_vector_type(4)));
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.y * blockDim.x + threadIdx.x;
  float2 d_gamma[LDG] = {0};
  float2 d_beta[LDG] = {0};
  
  __shared__ float shm[80];

  float c_h = float(0);
  at::BFloat16* c_h1 = reinterpret_cast<at::BFloat16*>(&c_h);
  float c_loss = float(0);
  at::BFloat16* c_loss1 = reinterpret_cast<at::BFloat16*>(&c_loss);
  float c_dres = float(0);
  at::BFloat16* c_dres1 = reinterpret_cast<at::BFloat16*>(&c_dres);
  float c_gamma = float(0);
  at::BFloat16* c_gamma1 = reinterpret_cast<at::BFloat16*>(&c_gamma);

  v4u32 gBase;
  gBase.x = (unsigned)(unsigned long long)gamma;
  gBase.y = (unsigned)((unsigned long long)gamma >> 32);
  gBase.zw = -1u;

  at::BFloat16 gamma_data[LDG*2];
  #pragma unroll
  for (int l=0;l<LDG;l++) {
    c_gamma = __ivcorex_ml_mem_load_f32(gBase, 4*(thrx+l*numx), 0, 0);
    gamma_data[l*2] = c_gamma1[0];
    gamma_data[l*2+1] = c_gamma1[1];
  }

  at::BFloat16 beta_data[LDG*2];
  if (MemoryEfficient && !RMSONLY) {
    float c_beta = float(0);
    at::BFloat16* c_beta1 = reinterpret_cast<at::BFloat16*>(&c_beta);

    v4u32 hBase;
    hBase.x = (unsigned)(unsigned long long)beta;
    hBase.y = (unsigned)((unsigned long long)beta >> 32);
    hBase.zw = -1u;

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_beta = __ivcorex_ml_mem_load_f32(hBase, 4*(thrx+l*numx), 0, 0);
      beta_data[l*2] = c_beta1[0];
      beta_data[l*2+1] = c_beta1[1];
    }
  }

  #pragma unroll
  for (int row = blockIdx.y;row<n1;row+=gridDim.y) {
    float sum_loss1 = float(0);
    float sum_loss2 = float(0);

    const float c_invvar = invvar[row];
    const float* input_or_output_ptr = reinterpret_cast<const float*>(input_or_output + row*n2);
    const float* dout_ptr = reinterpret_cast<const float*>(dout + row*n2);
    const float* dres_ptr = reinterpret_cast<const float*>(dres + row*n2);

    v4u32 aBase;
    aBase.x = (unsigned)(unsigned long long)input_or_output_ptr;
    aBase.y = (unsigned)((unsigned long long)input_or_output_ptr >> 32);
    aBase.zw = -1u;

    v4u32 bBase;
    bBase.x = (unsigned)(unsigned long long)dout_ptr;
    bBase.y = (unsigned)((unsigned long long)dout_ptr >> 32);
    bBase.zw = -1u;

    v4u32 cBase;
    cBase.x = (unsigned)(unsigned long long)dres_ptr;
    cBase.y = (unsigned)((unsigned long long)dres_ptr >> 32);
    cBase.zw = -1u;
    
    float2 y[LDG];
    float2 dy[LDG];

    #pragma unroll
    for (int l=0;l<LDG;l++) {
      c_h = __ivcorex_ml_mem_load_f32(aBase, 4 * (thrx+l*numx), 0, 0);
      c_loss = __ivcorex_ml_mem_load_f32(bBase, 4 * (thrx+l*numx), 0, 0);
      float y_tmp0;
      float y_tmp1;
      if (!RMSONLY) {
        if (!MemoryEfficient) {
          y_tmp0 = (static_cast<float>(c_h1[0]) - mean[row]) * c_invvar;
          y_tmp1 = (static_cast<float>(c_h1[1]) - mean[row]) * c_invvar;
        } else {
          y_tmp0 = (static_cast<float>(c_h1[0]) - static_cast<float>(beta_data[l*2])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = (static_cast<float>(c_h1[1]) - static_cast<float>(beta_data[l*2+1])) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      } else {
        if (!MemoryEfficient) {
          y_tmp0 = static_cast<float>(c_h1[0]) * c_invvar;
          y_tmp1 = static_cast<float>(c_h1[1]) * c_invvar;
        } else {
          y_tmp0 = static_cast<float>(c_h1[0]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2], eps));
          y_tmp1 = static_cast<float>(c_h1[1]) / static_cast<float>(clamp_by_magnitude(gamma_data[l*2+1], eps));
        }
      }
      float dy_tmp0 = static_cast<float>(c_loss1[0]) * static_cast<float>(gamma_data[l*2]);
      float dy_tmp1 = static_cast<float>(c_loss1[1]) * static_cast<float>(gamma_data[l*2+1]);
      sum_loss1 += dy_tmp0 + dy_tmp1;
      sum_loss2 += y_tmp0 * dy_tmp0 + y_tmp1 * dy_tmp1;
      y[l].x = y_tmp0;
      y[l].y = y_tmp1;
      dy[l].x = dy_tmp0;
      dy[l].y = dy_tmp1;
      d_gamma[l].x += static_cast<float>(c_loss1[0]) * y_tmp0;
      d_gamma[l].y += static_cast<float>(c_loss1[1]) * y_tmp1;
      if (!RMSONLY) {
        d_beta[l].x += static_cast<float>(c_loss1[0]);
        d_beta[l].y += static_cast<float>(c_loss1[1]);
      }
    }

    // intra warp reduction
    float val1;
    float val2;
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val1 = __shfl_xor_sync(0xffffffff, sum_loss1, offset, 64);
        val2 = __shfl_xor_sync(0xffffffff, sum_loss2, offset, 64);
        sum_loss1 += val1;
        sum_loss2 += val2;
    }

    // intra block reduction
    if (blockDim.y > 1) {
      int offset = 1<<(32 - __clz(blockDim.y-1) - 1);
      if (!RMSONLY) {
        if (threadIdx.x == 0) {
          shm[threadIdx.y*2] = sum_loss1;
          shm[threadIdx.y*2+1] = sum_loss2;
        }
        __syncthreads();
        for (;offset>0;offset/=2) {
          if (thrx < offset) {
            shm[thrx*2] += shm[thrx*2+offset*2];
            shm[thrx*2+1] += shm[thrx*2+offset*2+1];
          }
          __syncthreads();
        }
        sum_loss1 = shm[0];
        sum_loss2 = shm[1];
      } else {
        if (threadIdx.x == 0) {
          shm[threadIdx.y] = sum_loss2;
        }
        __syncthreads();
        #pragma unroll
        for (;offset>0;offset/=2) {
          if (thrx < offset && thrx+offset < blockDim.y) {
            shm[thrx] += shm[thrx+offset];
          }
          __syncthreads();
        }
        sum_loss2 = shm[0];
      }
    }

    float fH = (float)n2;
    float term1 = (float(1) / fH) * c_invvar;
    float* k_grad_input = reinterpret_cast<float*>(grad_input + row*n2);
    for (int l=0;l<LDG;l++) {
      float2 y_tmp;
      y_tmp.x = y[l].x;
      y_tmp.y = y[l].y;
      float2 dy_tmp;
      dy_tmp.x = dy[l].x;
      dy_tmp.y = dy[l].y;

      float f_grad_input0 = fH * dy_tmp.x;
      float f_grad_input1 = fH * dy_tmp.y;
      if (!RMSONLY) {
        f_grad_input0 -= sum_loss1 + y_tmp.x * sum_loss2;
        f_grad_input1 -= sum_loss1 + y_tmp.y * sum_loss2;
      } else {
        f_grad_input0 -= y_tmp.x * sum_loss2;
        f_grad_input1 -= y_tmp.y * sum_loss2;
      }
      f_grad_input0 *= term1;
      f_grad_input1 *= term1;

      c_dres = __ivcorex_ml_mem_load_f32(cBase, 4 * (thrx+l*numx), 0, 0);
      float f_grad_input;
      at::BFloat16* f_grad_input_ = reinterpret_cast<at::BFloat16*>(&f_grad_input);
      f_grad_input_[0] = static_cast<at::BFloat16>(f_grad_input0+static_cast<float>(c_dres1[0]));
      f_grad_input_[1] = static_cast<at::BFloat16>(f_grad_input1+static_cast<float>(c_dres1[1]));
      k_grad_input[thrx+l*numx] = f_grad_input;
    }
  }
  
  float2* part_grad_gamma_ptr = reinterpret_cast<float2*>(part_grad_gamma + blockIdx.y*n2);
  float2* part_grad_beta_ptr = reinterpret_cast<float2*>(part_grad_beta + blockIdx.y*n2);
  for (int l=0;l<LDG;l++) {
    part_grad_gamma_ptr[l*numx+thrx] = d_gamma[l];
    if (!RMSONLY) {
      part_grad_beta_ptr[l*numx+thrx] = d_beta[l];
    }
  }
}

template<bool RMSONLY, typename U, typename V> __global__
void ComputeGradGammaBeta_opt(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n2,
    V* grad_gamma,
    V* grad_beta)
{
  U tmp_gamma = U(0);
  U tmp_beta = U(0);
  int i2 = blockDim.x * blockIdx.x + threadIdx.x;
  #pragma unroll
  for (int k=0;k<part_size;++k) {
    tmp_gamma += part_grad_gamma[i2+k*n2];
    if (!RMSONLY) {
      tmp_beta += part_grad_beta[i2+k*n2];
    }
  }
  grad_gamma[i2] = static_cast<V>(tmp_gamma);
  if (!RMSONLY) {
    grad_beta[i2] = static_cast<V>(tmp_beta);
  }
}

template<typename T, typename U, typename V=T>
void HostApplyLayerNorm(
    V* output,
    U* mean,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    float epsilon,
    const V* gamma,
    const V* beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(32,4,1);
    dim3 threads(64,1,1);
    if (sizeof(T) == 1) {
      threads.y = n2/1024/4 > 1 ? n2/1024/4 : 1;
    }
    if (sizeof(T) == 2) {
      threads.y = n2/1024/2 > 1 ? n2/1024/2 : 1;
    }
    if (sizeof(T) >= 4) {
      threads.y = n2/1024 > 1 ? n2/1024 : 1;
    }
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY/threads.y), 1);
#else
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
      output, mean, invvar, input, n1, n2, U(epsilon), gamma, beta);
}

template<typename T, typename U, typename V=T>
void HostApplyRMSNorm(
    V* output,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const V* gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(32,4,1);
    dim3 threads(64,1,1);
    if (sizeof(T) == 1) {
      threads.y = n2/1024/4 > 1 ? n2/1024/4 : 1;
    }
    if (sizeof(T) == 2) {
      threads.y = n2/1024/2 > 1 ? n2/1024/2 : 1;
    }
    if (sizeof(T) >= 4) {
      threads.y = n2/1024 > 1 ? n2/1024 : 1;
    }
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY/threads.y), 1);
#else
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}

template<typename T, typename U, typename V=T>
void HostApplyRMSNormRes(
    V* output,
    V* sum,
    U* invvar,
    const T* input,
    const T* residual,
    int n1,
    int n2,
    double epsilon,
    const V* gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // const dim3 threads(32,4,1);
    dim3 threads(64,1,1);
    if (sizeof(T) == 1) {
      threads.y = n2/1024/4 > 1 ? n2/1024/4 : 1;
    }
    if (sizeof(T) == 2) {
      threads.y = n2/1024/2 > 1 ? n2/1024/2 : 1;
    }
    if (sizeof(T) >= 4) {
      threads.y = n2/1024 > 1 ? n2/1024 : 1;
    }
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY/threads.y), 1);
#else
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyRMSNormRes<<<blocks, threads, nshared, stream>>>(
      output, sum, invvar, input, residual, n1, n2, U(epsilon), gamma);
}


void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    float epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "layer_norm_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyLayerNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
              mean->DATA_PTR<accscalar_t>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
          beta != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL);
      )
}

void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL);
      )
}

void cuda_rms_norm_residual(
    at::Tensor* output,
    at::Tensor* sum,
    at::Tensor* invvar,
    at::Tensor* input,
    at::Tensor* residual,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_residual_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyRMSNormRes<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
          sum->DATA_PTR<scalar_t_out>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          residual->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL);
      )
}


template<typename T, typename U=float, typename V=T>
void HostLayerNormGradient(
    const V* dout,
    const U* mean,
    const U* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const V* gamma,
    const V* beta,
    float epsilon,
    T* grad_input,
    V* grad_gamma,
    V* grad_beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      // compute grad_gamma(j) and grad_beta(j)
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      // note (mkozuki): I can hard code part_grad_gamma's dtype as float given that
      // the `cuda_layer_norm_gradient` doesn't support double.
      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(part_grad_dtype));
      at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                      dout,
                      input->DATA_PTR<T>(),
                      n1,n2,
                      mean,
                      invvar,
                      U(epsilon),
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_beta.DATA_PTR<U>(),
                      false);

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_beta.DATA_PTR<U>(),
                      part_size,
                      n1,n2,
                      grad_gamma,
                      grad_beta,
                      false);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY/4), 1);
#else
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
    const dim3 threads1(32,4,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->DATA_PTR<T>(),
            n1,n2,
            mean,
            invvar,
            U(epsilon),
            gamma,
            grad_input,
            false);
}

template<typename T, typename U=float, typename V=T>
void HostRMSNormGradient(
    const V* dout,
    const U* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    V* grad_gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      // note (mkozuki): I can hard code part_grad_gamma's dtype as float given that
      // the `cuda_layer_norm_gradient` doesn't support double.
      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(part_grad_dtype));
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                      dout,
                      input->DATA_PTR<T>(),
                      n1,n2,
                      invvar, // unused
                      invvar,
                      U(epsilon),
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_gamma.DATA_PTR<U>(), /* unused */
                      true);

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_gamma.DATA_PTR<U>(), /* unused */
                      part_size,
                      n1,n2,
                      grad_gamma,
                      grad_gamma, /* unused */
                      true);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  #ifdef __ILUVATAR__
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY/4), 1);
#else
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
    const dim3 threads1(32,4,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->DATA_PTR<T>(),
            n1,n2,
            invvar, /* unused */
            invvar,
            U(epsilon),
            gamma,
            grad_input,
            true);
}

template<typename T, typename U=float, typename V=T>
void HostLayerNormGradient_opt(
    const V* dout,
    const U* mean,
    const U* invvar,
    at::Tensor* input, // max supported hidden size 64*32*40=81920
    int n1,
    int n2,
    const V* gamma,
    const V* beta,
    float epsilon,
    T* grad_input,
    V* grad_gamma,
    V* grad_beta,
    bool memory_efficient
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      int div = 1;
      if (sizeof(T) < 4) {div = 4/sizeof(T);}
      int blocky = n2/64/div;
      int LDG;
      int gridy;
      
      if (blocky > 32) {
        for (int i =2;i<blocky;i*=2) {
          if (blocky % i == 0 && blocky / i <= 40) {
            blocky /= i*div;
            break;
          }
        }
        gridy = 16 * 128 / blocky - 1;
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      } else {
        gridy = 16 * 128 / blocky;
        if (sizeof(T) == 2) {gridy -= 1;}
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      }
      LDG = n2/64/blocky/div;

      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({gridy,n2}, input->options().dtype(part_grad_dtype));
      at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);

      if (LDG == 1) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<1, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }
      if (LDG == 2) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<2, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }
      if (LDG == 4) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<4, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }
      if (LDG == 8) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<8, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }
      if (LDG == 16) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<16, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }
      if (LDG == 32) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<32, false, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, gamma, beta, double(epsilon), grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());});
      }


      const dim3 threads3 (64, 1, 1);
      const dim3 blocks3 (n2/64, 1, 1);
      ComputeGradGammaBeta_opt<false><<<blocks3, threads3, 0, stream>>>(
        part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>(), gridy, n2, grad_gamma, grad_beta
      );
    } else {
      // compute grad_input
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY/4), 1);
#else
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
      const dim3 threads1(32,4,1);
      int nshared =
              threads1.y > 1 ?
              threads1.y*threads1.x*sizeof(U) :
              0;
      cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
              dout,
              input->DATA_PTR<T>(),
              n1,n2,
              mean,
              invvar,
              U(epsilon),
              gamma,
              grad_input,
              false);
    }
}

template<typename T, typename U=float, typename V=T>
void HostRMSNormGradient_opt(
    const V* dout,
    const U* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    V* grad_gamma,
    bool memory_efficient)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      int div = 1;
      if (sizeof(T) < 4) {div = 4/sizeof(T);}
      int blocky = n2/64/div;
      int LDG;
      int gridy;
      
      if (blocky > 32) {
        for (int i =2;i<blocky;i*=2) {
          if (blocky % i == 0 && blocky / i <= 40) {
            blocky /= i*div;
            break;
          }
        }
        gridy = 16 * 128 / blocky - 1;
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      } else {
        gridy = 16 * 128 / blocky;
        if (sizeof(T) == 2) {gridy -= 1;}
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      }
      LDG = n2/64/blocky/div;

      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({gridy,n2}, input->options().dtype(part_grad_dtype));

      if (LDG == 1) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<1, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 2) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<2, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 4) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<4, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 8) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<8, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 16) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<16, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 32) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights<32, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }


      const dim3 threads3 (64, 1, 1);
      const dim3 blocks3 (n2/64, 1, 1);
      ComputeGradGammaBeta_opt<true><<<blocks3, threads3, 0, stream>>>(
        part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>(), gridy, n2, grad_gamma, grad_gamma
      );
    } else {

      // compute grad_input
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY/4), 1);
#else
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
      const dim3 threads1(32,4,1);
      int nshared =
              threads1.y > 1 ?
              threads1.y*threads1.x*sizeof(U) :
              0;
      cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
              dout,
              input->DATA_PTR<T>(),
              n1,n2,
              invvar, /* unused */
              invvar,
              U(epsilon),
              gamma,
              grad_input,
              true);
    }
}

template<typename T, typename U=float, typename V=T>
void HostRMSNormGradient_opt2(
    const V* dout,
    const V* dres,
    const U* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    V* grad_gamma,
    bool memory_efficient)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      int div = 1;
      if (sizeof(T) < 4) {div = 4/sizeof(T);}
      int blocky = n2/64/div;
      int LDG;
      int gridy;
      
      if (blocky > 32) {
        for (int i =2;i<blocky;i*=2) {
          if (blocky % i == 0 && blocky / i <= 40) {
            blocky /= i*div;
            break;
          }
        }
        gridy = 16 * 128 / blocky - 1;
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      } else {
        gridy = 16 * 128 / blocky;
        if (sizeof(T) == 2) {gridy -= 1;}
        for (int i = gridy;i>0;i--) {
          if (n1 % i == 0) {gridy = i;break;}
        }
      }
      LDG = n2/64/blocky/div;

      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({gridy,n2}, input->options().dtype(part_grad_dtype));

      if (LDG == 1) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<1, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 2) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<2, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 4) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<4, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 8) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<8, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 16) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<16, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }
      if (LDG == 32) {
        const dim3 threads2 (64, blocky, 1);
        const dim3 blocks2 (1, gridy, 1);
        BOOL_SWITCH(memory_efficient, MemoryEfficient, [&]{fusedGradInputWeights_<32, true, MemoryEfficient><<<blocks2, threads2, 0, stream>>>(dout, dres, input->DATA_PTR<T>(), n1, n2, invvar, invvar, gamma, gamma, epsilon, grad_input, part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>());});
      }


      const dim3 threads3 (64, 1, 1);
      const dim3 blocks3 (n2/64, 1, 1);
      ComputeGradGammaBeta_opt<true><<<blocks3, threads3, 0, stream>>>(
        part_grad_gamma.DATA_PTR<U>(), part_grad_gamma.DATA_PTR<U>(), gridy, n2, grad_gamma, grad_gamma
      );
    } else {

      // compute grad_input
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
#ifdef __ILUVATAR__
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY/4), 1);
#else
      const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
#endif
      const dim3 threads1(32,4,1);
      int nshared =
              threads1.y > 1 ?
              threads1.y*threads1.x*sizeof(U) :
              0;
      cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
              dout,
              input->DATA_PTR<T>(),
              n1,n2,
              invvar, /* unused */
              invvar,
              U(epsilon),
              gamma,
              grad_input,
              true);
    }
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    float epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta,
    bool memory_efficient)
{
    using namespace at;
    // we can do away with `accscalar_t` as there're only three dtypes: fp32, fp16, bf16
    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input->scalar_type(), gamma == NULL ? input->scalar_type() :  gamma->scalar_type(), "cuComputeGradInput",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostLayerNormGradient_opt(
        dout->DATA_PTR<scalar_t_out>(),
        mean != NULL ? mean->DATA_PTR<accscalar_t>() : NULL,
        invvar->DATA_PTR<accscalar_t>(),
        input,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        gamma != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
        gamma != NULL ? grad_beta->DATA_PTR<scalar_t_out>() : NULL,
        memory_efficient);
    )
}

void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient)
{
    using namespace at;
    // we can do away with `accscalar_t` as there're only three dtypes: fp32, fp16, bf16
    // DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input->scalar_type(), gamma == NULL ? input->scalar_type() :  gamma->scalar_type(), "cuComputeGradInputRMS",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostRMSNormGradient_opt(
        dout->DATA_PTR<scalar_t_out>(),
        invvar->DATA_PTR<accscalar_t>(),
        input,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
        memory_efficient);
    )
}

void cuda_rms_norm_residual_gradient(
    at::Tensor* dout,
    at::Tensor* dres,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    bool memory_efficient)
{
    using namespace at;
    // we can do away with `accscalar_t` as there're only three dtypes: fp32, fp16, bf16
    // DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input->scalar_type(), gamma == NULL ? input->scalar_type() :  gamma->scalar_type(), "cuComputeGradInputRMS",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostRMSNormGradient_opt2(
        dout->DATA_PTR<scalar_t_out>(),
        dres->DATA_PTR<scalar_t_out>(),
        invvar->DATA_PTR<accscalar_t>(),
        input,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
        memory_efficient);
    )
}
