#pragma once
#ifndef FWD_GN_KERNEL_H
#define FWD_GN_KERNEL_H

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
    T *rstd_data);

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
      T *dbias_data);

#endif
