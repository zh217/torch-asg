#include <torch/torch.h>

#include <iostream>
#include <vector>

at::Tensor d_sigmoid(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}


std::vector<at::Tensor>
fac_forward(
        at::Tensor inputs,
        at::Tensor targets,
        at::Tensor transition
) {

    auto alpha = at::zeros_like(inputs);
    auto trans_next = at::zeros_like(inputs);
    auto trans_self = at::zeros_like(inputs);
    return {alpha};
}


void fac_backward() {

}


void fcc_forward() {

}

void fcc_backward() {

}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fac_forward", &fac_forward, "FAC forward");
}
#endif