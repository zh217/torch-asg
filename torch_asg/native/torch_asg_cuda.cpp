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

using IntArrayRef = at::ArrayRef<int64_t>;

IntArrayRef _convert_to_array_ref(const at::Tensor &t) {
    return IntArrayRef{t.data<int64_t>(), static_cast<size_t>(t.numel())};
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fac_loss_gpu_template(
        const at::Tensor &transition, // num_labels * num_labels, transition[i][j] is transition from j to i
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const std::string &scale_mode) {
    return {};
}

std::vector<at::Tensor> fac_loss_gpu(const at::Tensor &transition,
                                     const at::Tensor &inputs,
                                     const at::Tensor &targets,
                                     const at::Tensor &input_lengths, // batch_size
                                     const at::Tensor &target_lengths, // batch_size
                                     const std::string &scale_mode
) {
    at::Tensor input_lengths_ = input_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    at::Tensor target_lengths_ = target_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    return AT_DISPATCH_FLOATING_TYPES(inputs.type(), "fac_loss_gpu", [&] {
        if (targets.scalar_type() == at::kLong) {
            return fac_loss_gpu_template<scalar_t, at::kLong>(transition, inputs, targets,
                                                              _convert_to_array_ref(input_lengths_),
                                                              _convert_to_array_ref(target_lengths_),
                                                              scale_mode);
        } else {
            return fac_loss_gpu_template<scalar_t, at::kInt>(transition, inputs, targets,
                                                             _convert_to_array_ref(input_lengths_),
                                                             _convert_to_array_ref(target_lengths_),
                                                             scale_mode);
        }
    });
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fac_loss_backward_gpu_template(
        const at::Tensor &grad_out,
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const at::Tensor &alpha,
        const at::Tensor &scale,
        const at::Tensor &self_trans,
        const at::Tensor &next_trans
) {
    return {};
}

std::vector<at::Tensor> fac_loss_backward_gpu(const at::Tensor &grad_out,
                                              const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
                                              const at::Tensor &targets, // batch_size * target_len
                                              const at::Tensor &input_lengths, // batch_size
                                              const at::Tensor &target_lengths, // batch_size
                                              const at::Tensor &alpha,
                                              const at::Tensor &scale,
                                              const at::Tensor &self_trans,
                                              const at::Tensor &next_trans
) {
    at::Tensor input_lengths_ = input_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    at::Tensor target_lengths_ = target_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    return AT_DISPATCH_FLOATING_TYPES(inputs.type(), "fac_loss_backward_gpu", [&] {
        if (targets.scalar_type() == at::kLong) {
            return fac_loss_backward_gpu_template<scalar_t, at::kLong>(grad_out, inputs, targets,
                                                                       _convert_to_array_ref(input_lengths_),
                                                                       _convert_to_array_ref(target_lengths_),
                                                                       alpha, scale, self_trans, next_trans);
        } else {
            return fac_loss_backward_gpu_template<scalar_t, at::kInt>(grad_out, inputs, targets,
                                                                      _convert_to_array_ref(input_lengths_),
                                                                      _convert_to_array_ref(target_lengths_),
                                                                      alpha, scale, self_trans,
                                                                      next_trans);
        }
    });
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fcc_loss_gpu_template(
        const at::Tensor &transition, // num_labels * num_labels, transition[i][j] is transition from j to i
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const std::string &scale_mode
) {
    return {};
}

std::vector<at::Tensor> fcc_loss_gpu(
        const at::Tensor &transition,
        const at::Tensor &inputs,
        const at::Tensor &targets,
        const at::Tensor &input_lengths, // batch_size
        const at::Tensor &target_lengths, // batch_size
        const std::string &scale_mode
) {
    at::Tensor input_lengths_ = input_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    at::Tensor target_lengths_ = target_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    return AT_DISPATCH_FLOATING_TYPES(inputs.type(), "fcc_loss_gpu", [&] {
        if (targets.scalar_type() == at::kLong) {
            return fcc_loss_gpu_template<scalar_t, at::kLong>(transition, inputs, targets,
                                                              _convert_to_array_ref(input_lengths_),
                                                              _convert_to_array_ref(target_lengths_),
                                                              scale_mode);
        } else {
            return fcc_loss_gpu_template<scalar_t, at::kInt>(transition, inputs, targets,
                                                             _convert_to_array_ref(input_lengths_),
                                                             _convert_to_array_ref(target_lengths_),
                                                             scale_mode);
        }
    });
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fcc_loss_backward_gpu_template(
        const at::Tensor &grad_out,
        const at::Tensor &transition,
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const at::Tensor &alpha,
        const at::Tensor &alpha_max_contrib,
        const at::Tensor &scale
) {
    return {};
}

std::vector<at::Tensor> fcc_loss_backward_gpu(
        const at::Tensor &grad_out,
        const at::Tensor &transition,
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        const at::Tensor &input_lengths, // batch_size
        const at::Tensor &target_lengths, // batch_size
        const at::Tensor &alpha,
        const at::Tensor &alpha_max_contrib,
        const at::Tensor &scale
) {
    at::Tensor input_lengths_ = input_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    at::Tensor target_lengths_ = target_lengths.toType(at::kLong).toBackend(at::Backend::CPU).contiguous();
    return AT_DISPATCH_FLOATING_TYPES(inputs.type(), "fcc_loss_backward_gpu", [&] {
        if (targets.scalar_type() == at::kLong) {
            return fcc_loss_backward_gpu_template<scalar_t, at::kLong>(grad_out, transition, inputs, targets,
                                                                       _convert_to_array_ref(input_lengths_),
                                                                       _convert_to_array_ref(target_lengths_),
                                                                       alpha, alpha_max_contrib, scale);
        } else {
            return fcc_loss_backward_gpu_template<scalar_t, at::kInt>(grad_out, transition, inputs, targets,
                                                                      _convert_to_array_ref(input_lengths_),
                                                                      _convert_to_array_ref(target_lengths_),
                                                                      alpha, alpha_max_contrib, scale);
        }
    });
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