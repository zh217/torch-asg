//
// Created by amade on 4/2/2019.
//
#include "fully_connected_lattice.h"
//#include <omp.h>

namespace torch_asg {

std::tuple<at::Tensor, at::Tensor>
fully_connected_alpha_recursion(
        at::Tensor &inputs,
        at::Tensor &transition,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();
    auto transition_e = transition.view({1, num_labels, num_labels}).contiguous();
    auto alpha = at::full({batch_input_len, num_batches, num_labels}, neg_inf,
                          inputs.options().requires_grad(false));
    auto path_contrib = at::zeros({batch_input_len - 1, num_batches, num_labels, num_labels},
                                  inputs.options().requires_grad(false));

    alpha[0] = inputs[0];

    for (int64_t t = 1; t < batch_input_len; ++t) {

        auto tmp = transition_e + inputs[t].view({num_batches, num_labels, 1}) +
                   alpha[t - 1].view({num_batches, 1, num_labels});
        path_contrib[t - 1] = tmp;
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
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();
    auto transition_t = transition.t().view({1, num_labels, num_labels}).contiguous();
    auto beta = at::full({batch_input_len, num_batches, num_labels}, neg_inf, inputs.options().requires_grad(false));

    beta[batch_input_len - 1].fill_(0);

    for (int64_t t = batch_input_len - 2; t >= 0; --t) {
        beta[t] = (transition_t + (inputs[t + 1] + beta[t + 1]).view({num_batches, 1, num_labels})).logsumexp(2);
    }

    return beta;
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

    bool should_roll = should_roll_to_end(input_lengths, batch_input_len);

    auto alpha_results = fully_connected_alpha_recursion(inputs, transition, batch_input_len, num_batches, num_labels);
    auto alpha = std::get<0>(alpha_results);
    auto path_contrib = std::get<1>(alpha_results);

//    at::Tensor forward_scores = at::zeros({num_batches}, inputs.options());
//    if (should_roll) {
//
//        for (int64_t b = 0; b < num_batches; ++b) {
//            forward_scores[b] = alpha[input_lengths[b] - 1][b].logsumexp(0);
//        }
//
//    } else {
//        forward_scores.copy_(alpha[batch_input_len - 1].logsumexp(1));
//    }

    at::Tensor input_aligned = should_roll ? roll_to_end(inputs, input_lengths) : inputs;
    auto beta = fully_connected_beta_recursion(input_aligned, transition, batch_input_len, num_batches, num_labels);
    beta = should_roll ? roll_to_end(beta, input_lengths, true) : beta;
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