//
// Created by amade on 4/2/2019.
//

#ifndef TORCH_ASG_FORCE_ALIGNED_LATTICE_H
#define TORCH_ASG_FORCE_ALIGNED_LATTICE_H

#include "utils.h"

namespace torch_asg {

void force_aligned_alpha_recursion(
        at::Tensor &alpha,
        at::Tensor &path_contrib,
        at::Tensor &aligned_inputs, // input_len, batch, output_len
        at::Tensor &aligned_transition, // 2, batch, output_len - 1
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
);

void force_aligned_beta_recursion(
        at::Tensor &beta,
        at::Tensor &aligned_inputs, // input_len, batch, output_len, has already been rolled
        at::Tensor &aligned_transition, // 2, batch, output_len
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
);

std::tuple<at::Tensor, at::Tensor>
force_aligned_derivative(
        at::Tensor &grad_out, // num_batches
        at::Tensor &gamma, // batch_input_len, num_batches, batch_output_len
        at::Tensor &path_contrib, // batch_input_len - 1, 2, num_batches, batch_output_len - 1
        int64_t num_batches,
        int64_t batch_output_len
);


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
force_aligned_forward(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &transition,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels,
        int64_t batch_output_len
);


std::tuple<at::Tensor, at::Tensor>
force_aligned_backward(
        at::Tensor &grad_out,
        at::Tensor &alpha,
        at::Tensor &beta,
        at::Tensor &path_contrib,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels,
        int64_t batch_output_len
);


}

#endif //TORCH_ASG_FORCE_ALIGNED_LATTICE_H
