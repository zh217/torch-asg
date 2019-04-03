//
// Created by amade on 4/2/2019.
//
#include <omp.h>
#include <torch/extension.h>

namespace torch_asg {

inline at::Tensor masked_softmax(at::Tensor &input, int64_t dim) {
    auto output = input.softmax(dim);
    // this is to deal with exp(-inf) / (exp(-inf) + exp(-inf)) = 0 / 0
    // the current version of ATen somehow doesn't have at::isnan()
    output.masked_fill_(output != output, 0);
    return output;
}

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
    auto path_contrib = at::empty({batch_input_len - 1, num_batches, num_labels, num_labels}, inputs.options());

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
        at::Tensor &grad_out,
        at::Tensor &gamma,
        at::Tensor &path_contrib,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    auto grad_inputs = gamma.softmax(2) * grad_out.view({1, num_batches, 1});
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

    bool should_roll = false;
    if (input_lengths.dim() > 0) {
        AT_ASSERT(input_lengths.dtype() == at::kLong)
        AT_ASSERT(input_lengths.size(0) == num_batches)
        auto input_lengths_a = input_lengths.accessor<int64_t, 1>();
        for (int64_t b = 0; b < num_batches; ++b) {
            if (input_lengths_a[b] != batch_input_len) {
                should_roll = true;
                break;
            }
        }
    }

    auto alpha_results = fully_connected_alpha_recursion(inputs, transition, batch_input_len, num_batches, num_labels);
    auto alpha = std::get<0>(alpha_results);
    auto path_contrib = std::get<1>(alpha_results);

    at::Tensor forward_scores = at::empty({num_batches}, inputs.options());
    if (should_roll) {
        for (int64_t b = 0; b < num_batches; ++b) {
            forward_scores[b] = alpha[input_lengths[b]][b].logsumexp(0);
        }
    } else {
        forward_scores.copy_(alpha[batch_input_len - 1].logsumexp(1));
    }

    at::Tensor input_aligned = should_roll ? roll_to_end(inputs, input_lengths) : inputs;
    auto beta = fully_connected_beta_recursion(input_aligned, transition, batch_input_len, num_batches, num_labels);
    beta = should_roll ? roll_to_end(beta, input_lengths, true) : beta;

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

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fully_connected_forward", &torch_asg::fully_connected_forward, "fully connected forward");
    m.def("fully_connected_backward", &torch_asg::fully_connected_backward, "fully connected backward");
}
#endif