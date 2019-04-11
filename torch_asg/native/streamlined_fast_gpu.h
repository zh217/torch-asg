//
// Created by amade on 4/11/2019.
//

#ifndef TORCH_ASG_STREAMLINED_FAST_GPU_H
#define TORCH_ASG_STREAMLINED_FAST_GPU_H


//
// Created by amade on 4/11/2019.
//

#include "utils.h"

namespace torch_asg {

at::Tensor
fast_asg_gpu_forward_only(
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

std::tuple<
        at::Tensor, // full_scores
        at::Tensor, // aligned_scores
        at::Tensor, // full_gamma
        at::Tensor, // aligned_gamma
        at::Tensor, // full_path_contrib
        at::Tensor  // aligned_path_contrib
>
fast_asg_gpu_forward(
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

std::tuple<
        at::Tensor, // grad_transition
        at::Tensor  // grad_inputs
>
fast_asg_gpu_backward(
        at::Tensor &grad_out_full,
        at::Tensor &grad_out_aligned,
        at::Tensor &gamma_full,
        at::Tensor &gamma_aligned,
        at::Tensor &path_contrib_full,
        at::Tensor &path_contrib_aligned,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels,
        int64_t batch_output_len
);

}

#endif //TORCH_ASG_STREAMLINED_FAST_GPU_H
