//
// Created by amade on 4/3/2019.
//

#ifndef TORCH_ASG_UTILS_H
#define TORCH_ASG_UTILS_H

#include <torch/extension.h>

namespace torch_asg {

at::Tensor
masked_softmax(
        at::Tensor &input,
        int64_t dim
);

bool
should_roll_to_end(
        at::Tensor &input_lengths,
        int64_t batch_input_len
);

at::Tensor
roll_to_end(
        at::Tensor &aligned,
        at::Tensor &input_lengths,
        bool to_front = false
);

}

#define MY_DISPATCH_FLOAT(func, fst_arg, ...) \
(fst_arg.dtype() == at::kFloat) ? func<float>(fst_arg, ##__VA_ARGS__) : func<double>(fst_arg, ##__VA_ARGS__)

#endif //TORCH_ASG_UTILS_H
