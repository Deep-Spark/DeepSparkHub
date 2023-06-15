#ifndef CUDA_IMPL_YOLOV5_DECODE_KERNEL_H_
#define CUDA_IMPL_YOLOV5_DECODE_KERNEL_H_

#include <cuda.h>

template <typename T>
inline __device__ T devExp(T x) { return expf(x); }

template <typename T>
inline __device__ T devSigmoid(T x) { return 1.0f / (1.0f + devExp(-x)); }


/* pred_map: (batch_size, height, width, num_anchors, 5+num_classes) */
template <typename T>
__global__ void YoloV5DecodeForwardKernel(
    const T* pred_map,
    const T* anchors,
    T* decoded_bboxes,
    int stride,
    int batch_size,
    int height,
    int width,
    int num_classes,
    int num_anchors)
{
    int fm_area = height * width;
    int num_attrib = 5 + num_classes;
    int total_bboxes = batch_size * fm_area * num_anchors;

    int thd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thd_idx >= total_bboxes)
        return;

    // 计算某个维度的 idx 时， 先把该维度变成 inner most dim（除以其右边的所有维度的 dim 值的积）
    // 然后对该维度的 dim 值取余（避免 idx 超过 dim 值）
    //int batch_idx = thd_idx / (fm_area * num_anchors) % batch_size;
    int anchor_idx = thd_idx / fm_area % num_anchors;
    int h_idx = thd_idx / width % height;
    int w_idx = thd_idx % width;

    T* pred_attribs = (T*)(pred_map + thd_idx * num_attrib);

    T cx = (devSigmoid<T>(pred_attribs[0]) * 2 - 0.5 + w_idx) * stride;
    T cy = (devSigmoid<T>(pred_attribs[1]) * 2 - 0.5 + h_idx) * stride;
    T w_tmp = devSigmoid<T>(pred_attribs[2]) * 2;
    T w = w_tmp * w_tmp * anchors[2*anchor_idx] * stride;
    T h_tmp = devSigmoid<T>(pred_attribs[3]) * 2;
    T h = h_tmp * h_tmp * anchors[2*anchor_idx+1] * stride;
    T conf = devSigmoid<T>(pred_attribs[4]);

    int class_id = 0;
    T max_prob = 0.0;
    T* pred_probs = (T*)(pred_attribs + 5);
#pragma unroll
    for (int k = 0; k < num_classes; ++k) {
        T prob = devSigmoid<T>(pred_probs[k]);
        if (prob > max_prob) {
            max_prob = prob;
            class_id = k + 1;
        }
    }
    T* decoded_bbox = (T*)(decoded_bboxes + thd_idx * 6);
    T x1 = cx - 0.5 * w;
    T y1 = cy - 0.5 * h;
    decoded_bbox[0] = x1;
    decoded_bbox[1] = y1;
    decoded_bbox[2] = x1 + w;
    decoded_bbox[3] = y1 + h;
    decoded_bbox[4] = max_prob * conf;
    decoded_bbox[5] = (T)class_id;
}

#endif // CUDA_IMPL_YOLOV5_DECODE_KERNEL_H_
