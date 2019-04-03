//
// Created by amade on 4/3/2019.
//

#include "utils.h"


namespace torch_asg {


at::Tensor
masked_softmax(at::Tensor &input, int64_t dim) {
//    constexpr auto neg_inf = -std::numeric_limits<double>::max();
    auto output = input.softmax(dim);
    // this is to deal with exp(-inf) / (exp(-inf) + exp(-inf)) = 0 / 0
    // the current version of ATen somehow doesn't have at::isnan()
    // and picking up -inf from the tensor also seems problematic from the C++ API
    output.masked_fill_(output != output, 0);
    return output;
}

bool
should_roll_to_end(
        at::Tensor &input_lengths,
        int64_t batch_input_len
) {
    if (input_lengths.dim() > 0) {
        AT_ASSERT(input_lengths.dtype() == at::kLong)
        auto num_batches = input_lengths.size(0);
        auto input_lengths_a = input_lengths.accessor<int64_t, 1>();
        for (int64_t b = 0; b < num_batches; ++b) {
            if (input_lengths_a[b] != batch_input_len) {
                return true;
            }
        }
    }
    return false;
}

at::Tensor
roll_to_end(
        at::Tensor &aligned, // inp_len, batch, xxx
        at::Tensor &input_lengths,
        bool to_front
) {
    AT_ASSERT(input_lengths.dtype() == at::kLong);
    constexpr auto neg_inf = -std::numeric_limits<double>::max();
    auto rolled = at::full_like(aligned, neg_inf);
    auto aligned_transposed = aligned.permute({1, 0, 2});
    auto rolled_transposed = rolled.permute({1, 0, 2});
    auto n_batch = input_lengths.size(0);
    auto batch_input_len = aligned.size(0);
    auto input_lengths_a = input_lengths.accessor<int64_t, 1>();

#pragma omp parallel
    for (int64_t b = 0; b < n_batch; ++b) {
        auto inp_len = input_lengths_a[b];
        if (to_front) {
            rolled_transposed[b].slice(0, 0, inp_len) =
                    aligned_transposed[b].slice(0, batch_input_len - inp_len, batch_input_len);
        } else {
            rolled_transposed[b].slice(0, batch_input_len - inp_len, batch_input_len) =
                    aligned_transposed[b].slice(0, 0, inp_len);
        }
    }
    return rolled_transposed;
}

}
