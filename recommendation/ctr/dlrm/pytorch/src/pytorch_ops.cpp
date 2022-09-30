#include <torch/extension.h>

// torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input, torch::Tensor bottom_mlp_output, uint pad);
// std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input, torch::Tensor upstreamGrad, uint pad);
torch::Tensor gather_gpu_fwd(torch::Tensor input, torch::Tensor weight);
torch::Tensor gather_gpu_bwd(const torch::Tensor grad, const torch::Tensor indices, const int num_features);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("dotBasedInteractFwd", &dotBasedInteractFwdTorch, "", py::arg("input"), py::arg("bottom_mlp_output"), py::arg("pad"));
  // m.def("dotBasedInteractBwd", &dotBasedInteractBwdTorch, "", py::arg("input"), py::arg("upstreamGrad"), py::arg("pad"));
  m.def("gather_gpu_fwd", &gather_gpu_fwd, "Embedding gather", py::arg("indices"), py::arg("weight"));
  m.def("gather_gpu_bwd", &gather_gpu_bwd, "Embedding gather backward",
        py::arg("grad"), py::arg("indices"), py::arg("num_features"));
}
