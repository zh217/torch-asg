//
// Created by amade on 4/2/2019.
//

#ifndef TORCH_ASG_FORCE_ALIGNED_LATTICE_H
#define TORCH_ASG_FORCE_ALIGNED_LATTICE_H

#include "utils.h"

namespace torch_asg {


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
