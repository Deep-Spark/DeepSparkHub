#include <c10/util/Optional.h>
#include <torch/extension.h>

#include "swiglu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_fwd", &launch_swiglu_kernel, "");
    m.def("swiglu_bwd", &launch_swiglu_kernel_bwd, "");
    
}
