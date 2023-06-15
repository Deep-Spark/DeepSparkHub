#include "pytorch_cuda_helper.h"
#include "yolov5_decode_kernel.cuh"


namespace cudaImpl {

namespace {

at::Tensor YoloV5DecodeForwardKernelLauncher(
    const at::Tensor& pred_map,
    const at::Tensor& anchors,
    int64_t stride)
{
    TORCH_CHECK(pred_map.is_cuda(), "pred_map must be a CUDA tensor");
    TORCH_CHECK(anchors.is_cuda(), "anchors must be a CUDA tensor");

    int batch_size = pred_map.size(0);
    int num_anchors = pred_map.size(1);
    int height = pred_map.size(2);
    int width = pred_map.size(3);
    int num_classes = pred_map.size(4) - 5;
    int total_bboxes = batch_size * height * width * num_anchors;

    at::cuda::CUDAGuard device_guard(pred_map.device());

    TORCH_CHECK(pred_map.is_contiguous() && anchors.is_contiguous(),
                "Tensor must be contiguous")

    at::Tensor decoded_bboxes = at::zeros(
        {batch_size, int(height*width*num_anchors), 6}, pred_map.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        pred_map.scalar_type(), "YoloV5DecodeForwardKernel", [&] {
            YoloV5DecodeForwardKernel<scalar_t>
              <<<GET_BLOCKS(total_bboxes), THREADS_PER_BLOCK, 0, stream>>>(
                  pred_map.data_ptr<scalar_t>(),
                  anchors.data_ptr<scalar_t>(),
                  decoded_bboxes.data_ptr<scalar_t>(),
                  stride,
                  batch_size,
                  height,
                  width,
                  num_classes,
                  num_anchors);
        });
    AT_CUDA_CHECK(cudaGetLastError());
    return decoded_bboxes;
}

} // namespace

TORCH_LIBRARY_IMPL(cudaImpl, CUDA, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("cudaImpl::yolov5_decode_forward"),
        TORCH_FN(YoloV5DecodeForwardKernelLauncher));
    }

} // namespace cudaImpl
