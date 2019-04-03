//
// Created by amade on 4/2/2019.
//

#ifndef TORCH_ASG_FULLY_CONNECTED_LATTICE_H
#define TORCH_ASG_FULLY_CONNECTED_LATTICE_H

#include "utils.h"

namespace torch_asg {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fully_connected_forward(
        at::Tensor &inputs,
        at::Tensor &transition,
        at::Tensor &input_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
);

std::tuple<at::Tensor, at::Tensor>
fully_connected_backward(
        at::Tensor &grad_out,
        at::Tensor &alpha,
        at::Tensor &beta,
        at::Tensor &path_contrib,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
);

}

#endif //TORCH_ASG_FULLY_CONNECTED_LATTICE_H
