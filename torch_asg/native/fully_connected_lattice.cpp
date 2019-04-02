//
// Created by amade on 4/2/2019.
//
#include <omp.h>
#include <torch/extension.h>

namespace torch_asg {

at::Tensor
roll_to_end(
        at::Tensor &aligned, // inp_len, batch, xxx
        at::Tensor &input_lengths,
        bool to_front = false
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


std::tuple<at::Tensor, at::Tensor>
fully_connected_alpha_recursion(
        at::Tensor &inputs,
        at::Tensor &transition,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto transition_e = transition.view({1, num_labels, num_labels}).contiguous();
    auto alpha = at::empty({batch_input_len, num_batches, num_labels}, inputs.options());
    auto path_contrib = at::empty({batch_input_len, num_batches, num_labels, num_labels}, inputs.options());

    alpha[0] = inputs[0];

    for (int64_t t = 1; t < batch_input_len; ++t) {

        auto tmp = transition_e + inputs[t].view({num_batches, num_labels, 1}) +
                   alpha[t - 1].view({num_batches, 1, num_labels});
        path_contrib[t] = tmp;
        alpha[t] = tmp.logsumexp(2);
    }
    return {alpha, path_contrib};
}


at::Tensor
fully_connected_beta_recursion(
        at::Tensor &inputs,
        at::Tensor &transition,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto transition_t = transition.t().view({1, num_labels, num_labels}).contiguous();
    auto beta = at::empty({batch_input_len, num_batches, num_labels}, inputs.options());

    beta[batch_input_len - 1].fill_(0);

    for (int64_t t = batch_input_len - 2; t >= 0; --t) {
        beta[t] = (transition_t + (inputs[t + 1] + beta[t + 1]).view({num_batches, 1, num_labels})).logsumexp(2);
    }

    return beta;
}

std::tuple<at::Tensor, at::Tensor>
fully_connected_derivative(
        at::Tensor &alpha,
        at::Tensor &beta,
        at::Tensor &alpha_path_contrib,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto grad_inputs = (alpha + beta).softmax(2);
    auto grad_transition = (grad_inputs.view({batch_input_len, num_batches, num_labels, num_labels}) *
                            (alpha_path_contrib / alpha_path_contrib.sum(2, true))).sum({0, 1});

    return {grad_transition, grad_inputs};
}

void fully_connected_forward() {

}

void fully_connected_backward() {

}

}
