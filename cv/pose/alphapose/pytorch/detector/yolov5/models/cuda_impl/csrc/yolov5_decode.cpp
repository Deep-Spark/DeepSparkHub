#include <torch/extension.h>


namespace cudaImpl {

at::Tensor yolov5_decode_forward(
    const at::Tensor& pred_map,
    const at::Tensor& anchors,
    int64_t stride)
{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("cudaImpl::yolov5_decode_forward", "")
        .typed<decltype(yolov5_decode_forward)>();
    return op.call(
        pred_map, anchors, stride
    );
}

TORCH_LIBRARY_FRAGMENT(cudaImpl, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "cudaImpl::yolov5_decode_forward(Tensor pred_map, Tensor anchors, int stride) -> Tensor"));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("yolov5_decode_forward", &yolov5_decode_forward, "yolov5_decode_forward kernel warpper");
}

} // namespace cudaImpl
