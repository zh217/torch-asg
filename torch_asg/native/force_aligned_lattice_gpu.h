//
// Created by amade on 4/2/2019.
//

#ifndef TORCH_ASG_FORCE_ALIGNED_LATTICE_GPU_H
#define TORCH_ASG_FORCE_ALIGNED_LATTICE_GPU_H

#ifdef TORCH_ASG_SUPPORTS_CUDA

#include "force_aligned_lattice_kernel.h"

#else


namespace torch_asg {

template<typename scalar_t>
at::Tensor
make_aligned_inputs_gpu(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
) {
    throw std::runtime_error("Not compiled with CUDA support");
}

template<typename scalar_t>
at::Tensor
make_aligned_transition_gpu(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
) {
    throw std::runtime_error("Not compiled with CUDA support");
}

template<typename scalar_t>
void collect_transition_grad_gpu(
        at::Tensor &transition_grad,
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
) {
    throw std::runtime_error("Not compiled with CUDA support");
}

template<typename scalar_t>
void collect_input_grad_gpu(
        at::Tensor &input_grad,
        at::Tensor &aligned_input_grad,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    throw std::runtime_error("Not compiled with CUDA support");
}

}

#endif

#endif //TORCH_ASG_FORCE_ALIGNED_LATTICE_GPU_H
