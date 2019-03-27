//
// Created by amade on 3/27/2019.
//
#include <torch/extension.h>
//#include <ATen/ATen.h>
//#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

#include <numeric>
#include <type_traits>

#include <iostream>
#include <vector>
#include <limits>

namespace torch_asg {

void fac_loss_gpu() {

}

void fac_loss_backward_gpu() {

}

void fcc_loss_gpu() {

}

void fcc_loss_backward_gpu() {

}
}


#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fac_loss_gpu", &torch_asg::fac_loss_gpu, "FAC forward");
    m.def("fac_loss_backward_gpu", &torch_asg::fac_loss_backward_gpu, "FAC backward");
    m.def("fcc_loss_gpu", &torch_asg::fcc_loss_gpu, "FAC forward");
    m.def("fcc_loss_backward_gpu", &torch_asg::fcc_loss_backward_gpu, "FAC backward");
}
#endif