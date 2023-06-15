#ifndef CUDA_IMPL_COMMON_CUDA_HELPER_H_
#define CUDA_IMPL_COMMON_CUDA_HELPER_H_

#include <cuda.h>

#ifndef warpSize
#ifdef __ILUVATAR__
#define warpSize 64
#else
#define warpSize 32
#endif
#endif

#ifndef THREADS_PER_BLOCK
#ifdef __ILUVATAR__
#define THREADS_PER_BLOCK 4096
#else
#define THREADS_PER_BLOCK 1024
#endif
#endif

#ifndef MAX_BLOCKS
#define MAX_BLOCKS 1024
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
        i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
        j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
    for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
        for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
    int optimal_block_num = (N + num_threads - 1) / num_threads;
    return std::min(optimal_block_num, MAX_BLOCKS);
}

#endif // CUDA_IMPL_COMMON_CUDA_HELPER_H_
