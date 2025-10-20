#pragma once
#ifndef VECS_H
#define VECS_H

template <typename T, int num_elems>
struct float_vec;

template <typename T>
struct alignas(1 * sizeof(T)) float_vec<T, 1> {
  T x;
  template <typename U>
  __host__ __device__ operator float_vec<U, 1>() const {
      return { static_cast<U>(x), };
  }
};

template <typename T>
struct alignas(2 * sizeof(T)) float_vec<T, 2> {
  T x, y;
  template <typename U>
  __host__ __device__ operator float_vec<U, 2>() const {
      return { static_cast<U>(x), static_cast<U>(y), };
  }
};

template <typename T>
struct alignas(4 * sizeof(T)) float_vec<T, 4> {
  T x, y, z, w;
  template <typename U>
  __host__ __device__ operator float_vec<U, 4>() const {
      return { static_cast<U>(x), static_cast<U>(y), static_cast<U>(z), static_cast<U>(w), };
  }
};

#endif
