// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha);

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

std::vector<std::vector<int>> nms_match(Tensor dets, float iou_threshold);



void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width, float spatial_scale);

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height, int pooled_width,
                       float spatial_scale);

void sync_bn_forward_mean(const Tensor input, Tensor mean);

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var);

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size);

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_compiling_cuda_version", &get_compiling_cuda_version,
        "get_compiling_cuda_version");
  
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward,
        "sigmoid_focal_loss_forward ", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward,
        "sigmoid_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("grad_input"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward,
        "softmax_focal_loss_forward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward,
        "softmax_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("buff"), py::arg("grad_input"),
        py::arg("gamma"), py::arg("alpha"));
  m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("scores"),
        py::arg("iou_threshold"), py::arg("offset"));
  m.def("softnms", &softnms, "softnms (CPU) ", py::arg("boxes"),
        py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
        py::arg("sigma"), py::arg("min_score"), py::arg("method"),
        py::arg("offset"));
  m.def("nms_match", &nms_match, "nms_match (CPU) ", py::arg("dets"),
        py::arg("iou_threshold"));
  m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"), py::arg("argmax"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
        py::arg("grad_input"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"));
  m.def("sync_bn_forward_mean", &sync_bn_forward_mean, "sync_bn forward_mean",
        py::arg("input"), py::arg("mean"));
  m.def("sync_bn_forward_var", &sync_bn_forward_var, "sync_bn forward_var",
        py::arg("input"), py::arg("mean"), py::arg("var"));
  m.def("sync_bn_forward_output", &sync_bn_forward_output,
        "sync_bn forward_output", py::arg("input"), py::arg("mean"),
        py::arg("var"), py::arg("weight"), py::arg("bias"),
        py::arg("running_mean"), py::arg("running_var"), py::arg("norm"),
        py::arg("std"), py::arg("output"), py::arg("eps"), py::arg("momentum"),
        py::arg("group_size"));
  m.def("sync_bn_backward_param", &sync_bn_backward_param,
        "sync_bn backward_param", py::arg("grad_output"), py::arg("norm"),
        py::arg("grad_weight"), py::arg("grad_bias"));
  m.def("sync_bn_backward_data", &sync_bn_backward_data,
        "sync_bn backward_data", py::arg("grad_output"), py::arg("weight"),
        py::arg("grad_weight"), py::arg("grad_bias"), py::arg("norm"),
        py::arg("std"), py::arg("grad_input"));
}
