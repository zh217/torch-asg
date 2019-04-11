//
// Created by amade on 4/2/2019.
//
#include "fully_connected_lattice.h"
//#include <omp.h>

namespace torch_asg {

void fully_connected_alpha_recursion(
        at::Tensor &alpha,
        at::Tensor &path_contrib,
        at::Tensor &inputs,
        at::Tensor &transition,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto transition_e = transition.view({1, num_labels, num_labels}).contiguous();

    alpha[0] = inputs[0];

    for (int64_t t = 1; t < batch_input_len; ++t) {

        auto tmp = transition_e + inputs[t].view({num_batches, num_labels, 1}) +
                   alpha[t - 1].view({num_batches, 1, num_labels});
        path_contrib[t - 1] = tmp;
        alpha[t] = tmp.logsumexp(2);
    }
}


void fully_connected_beta_recursion(
        at::Tensor &beta,
        at::Tensor &inputs,
        at::Tensor &transition,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto transition_t = transition.t().view({1, num_labels, num_labels}).contiguous();

    beta[batch_input_len - 1].fill_(0);

    for (int64_t t = batch_input_len - 2; t >= 0; --t) {
        beta[t] = (transition_t + (inputs[t + 1] + beta[t + 1]).view({num_batches, 1, num_labels})).logsumexp(2);
    }
}

std::tuple<at::Tensor, at::Tensor>
fully_connected_derivative(
        at::Tensor &grad_out,
        at::Tensor &gamma,
        at::Tensor &path_contrib,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto grad_inputs = masked_softmax(gamma, 2) * grad_out.view({1, num_batches, 1});
    auto grad_transition = (grad_inputs.slice(0, 1).view({batch_input_len - 1, num_batches, num_labels, 1}) *
                            masked_softmax(path_contrib, 3)).sum({0, 1});

    return {grad_transition, grad_inputs};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fully_connected_forward(
        at::Tensor &inputs,
        at::Tensor &transition,
        at::Tensor &input_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();
    auto alpha = at::full({batch_input_len, num_batches, num_labels}, neg_inf,
                          inputs.options().requires_grad(false));
    auto path_contrib = at::zeros({batch_input_len - 1, num_batches, num_labels, num_labels},
                                  inputs.options().requires_grad(false));
    auto beta = at::full({batch_input_len, num_batches, num_labels}, neg_inf, inputs.options().requires_grad(false));

    at::Tensor input_lengths_cpu = input_lengths.is_cuda() ? input_lengths.to(at::kCPU, false, true) : input_lengths;
    bool should_roll = should_roll_to_end(input_lengths_cpu, batch_input_len);

    fully_connected_alpha_recursion(alpha, path_contrib, inputs, transition, batch_input_len, num_batches, num_labels);

    at::Tensor input_aligned = should_roll ? roll_to_end(inputs, input_lengths_cpu) : inputs;
    fully_connected_beta_recursion(beta, input_aligned, transition, batch_input_len, num_batches, num_labels);
    beta = should_roll ? roll_to_end(beta, input_lengths_cpu, true) : beta;
    auto forward_scores = (beta[0] + inputs[0]).logsumexp(1);
    return {forward_scores, alpha, beta, path_contrib};
}

std::tuple<at::Tensor, at::Tensor>
fully_connected_backward(
        at::Tensor &grad_out,
        at::Tensor &alpha,
        at::Tensor &beta,
        at::Tensor &path_contrib,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto gamma = alpha + beta;
    return fully_connected_derivative(grad_out, gamma, path_contrib, batch_input_len, num_batches, num_labels);
}

}