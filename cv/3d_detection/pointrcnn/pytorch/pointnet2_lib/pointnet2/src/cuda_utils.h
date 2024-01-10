#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <algorithm>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<float>(work_size)) / std::log(2.0);
    
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}
#endif
