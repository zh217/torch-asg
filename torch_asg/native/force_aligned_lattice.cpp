//
// Created by amade on 4/2/2019.
//

#include "force_aligned_lattice.h"

#include <omp.h>
#include <limits>

namespace torch_asg {


template<typename scalar_t>
at::Tensor
make_aligned_inputs(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
) {
    constexpr auto neg_inf = -std::numeric_limits<scalar_t>::max();
    at::Tensor aligned = at::full({batch_input_len, num_batches, batch_output_len}, neg_inf, inputs.options());
    auto aligned_a = aligned.accessor<scalar_t, 3>();
    auto inputs_a = inputs.accessor<scalar_t, 3>();
    auto outputs_a = outputs.accessor<int64_t, 2>();
    auto input_lengths_a = input_lengths.accessor<int64_t, 1>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();

#pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < num_batches; ++b) {
        for (int64_t t = 0; t < batch_input_len; ++t) {
            for (int64_t s = 0; s < batch_output_len; ++s) {
                if (t < input_lengths_a[b] && s < output_lengths_a[b]) {
                    aligned_a[t][b][s] = inputs_a[t][b][outputs_a[b][s]];
                }
            }
        }
    }
    return aligned;
}

template<typename scalar_t>
at::Tensor
make_aligned_transition(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
) {
//    constexpr auto neg_inf = -std::numeric_limits<scalar_t>::max();
    at::Tensor aligned = at::empty({2, num_batches, batch_output_len}, transition.options());
    auto transition_a = transition.accessor<scalar_t, 2>();
    auto aligned_a = aligned.accessor<scalar_t, 3>();
    auto outputs_a = outputs.accessor<int64_t, 2>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();

#pragma omp parallel
    for (int64_t b = 0; b < num_batches; ++b) {
        auto cur_output_len = output_lengths_a[b];
        for (int64_t s = 0; s < cur_output_len - 1; ++s) {
            auto cur = outputs_a[b][s];
            auto nxt = outputs_a[b][s + 1];
            aligned_a[0][b][s] = transition_a[cur][cur];
            aligned_a[1][b][s] = transition_a[nxt][cur];
        }
        auto last = outputs_a[b][cur_output_len - 1];
        aligned_a[0][b][cur_output_len - 1] = transition_a[last][last];
    }
    return aligned;
}

std::tuple<at::Tensor, at::Tensor>
force_aligned_alpha_recursion(
        at::Tensor &aligned_inputs, // input_len, batch, output_len
        at::Tensor &aligned_transition, // 2, batch, output_len - 1
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
) {
    constexpr auto neg_inf = -std::numeric_limits<double>::max();

    auto alpha = aligned_inputs.clone().detach();

    alpha[0].slice(1, 1).fill_(neg_inf);

    auto alpha_inv_idx = alpha.permute({2, 1, 0}); // output_len, num_batches, input_len
    auto aligned_inputs_inv_idx = aligned_inputs.permute({2, 1, 0});
    auto path_contrib = at::empty({batch_input_len - 1,
                                   2,
                                   num_batches,
                                   batch_output_len - 1}, aligned_inputs.options());
    auto self_transition = aligned_transition[0];
    auto next_transition = aligned_transition[1];

    alpha_inv_idx[0].slice(1, 1) += self_transition.permute({1, 0})[0].view({1, num_batches});
    alpha_inv_idx[0] = alpha_inv_idx[0].cumsum(1);

    auto alpha_no_top = alpha.slice(2, 1, batch_output_len);
    auto alpha_no_bottom = alpha.slice(2, 0, batch_output_len - 1);

    for (int64_t t = 1; t < batch_input_len; ++t) {
        path_contrib[t - 1][0] = alpha_no_top[t - 1] + self_transition.slice(1, 1, batch_output_len);
        path_contrib[t - 1][1] = alpha_no_bottom[t - 1] + next_transition.slice(1, 0, batch_output_len - 1);
        alpha_no_top[t] += path_contrib[t - 1].logsumexp(0);
    }

    return {alpha, path_contrib};
}

at::Tensor
force_aligned_beta_recursion(
        at::Tensor &aligned_inputs, // input_len, batch, output_len, has already been rolled
        at::Tensor &aligned_transition, // 2, batch, output_len
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
) {
    constexpr auto neg_inf = -std::numeric_limits<double>::max();

    at::Tensor beta = at::full_like(aligned_inputs, neg_inf); // input_len, batch, output_len

    std::cout << "A";

    auto self_transition = aligned_transition[0]; // batch, output_len
    auto next_transition = aligned_transition[1]; // batch, output_len

    std::cout << "B";

    for (int64_t b = 0; b < num_batches; ++b) {
        beta[batch_input_len - 1][b][output_lengths[b] - 1] = 1;
    }

    std::cout << "C";

    auto beta_last_row = aligned_inputs.permute({2, 0, 1})[batch_output_len - 1].slice(0, 1, batch_input_len)
                         // ^^ input_len, batch, remove input_len = 0
                         // vv output_len, batch -> batch -> 1, batch
                         + self_transition.permute({1, 0})[batch_input_len - 2].view({1, num_batches});
    std::cout << "Cx";
    beta_last_row = beta_last_row.flip(1);
    std::cout << "Cy";
    beta_last_row = beta_last_row.cumsum(1);
    std::cout << "Cz";
    beta_last_row = beta_last_row.flip(1);

    std::cout << "C1";
    // output_len, input_len, batch -> input_len, batch -> remove last idx
    beta.permute({2, 0, 1})[batch_output_len - 1].slice(0, 0, batch_input_len - 1) = beta_last_row;

    std::cout << "D";

    auto beta_no_top = beta.slice(2, 1, batch_output_len);
    auto beta_no_bottom = beta.slice(2, 0, batch_output_len - 1);

    std::cout << "E";

    for (int64_t t = batch_input_len - 2; t >= 0; --t) {
        beta_no_bottom[t] = at::stack(
                {self_transition.slice(1, 0, batch_output_len - 1)
                 + aligned_inputs[t + 1].slice(1, 0, batch_output_len - 1)
                 + beta_no_bottom[t + 1],
                 next_transition.slice(1, 0, batch_output_len - 1)
                 + aligned_inputs[t + 1].slice(1, 1, batch_output_len)
                 + beta_no_top[t + 1]},
                0).logsumexp(0);
    }

    std::cout << "F";

    return beta;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
force_aligned_derivative(
        at::Tensor &grad_out,
        at::Tensor &gamma,
        int64_t num_batches
) {
    auto grad_aligned_inputs = gamma.softmax(2) * grad_out.view({1, num_batches, 1});
    return {grad_out, grad_out, grad_out};
}

template<typename scalar_t>
at::Tensor
collect_scores(
        at::Tensor &alpha,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches
) {
    at::Tensor result = at::empty({num_batches}, alpha.options());
    auto alpha_a = alpha.accessor<scalar_t, 3>();
    auto result_a = result.accessor<scalar_t, 1>();
    auto input_lengths_a = input_lengths.accessor<int64_t, 1>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();
    for (int64_t b = 0; b < num_batches; ++b) {
        result_a[b] = alpha_a[input_lengths_a[b] - 1][b][output_lengths_a[b] - 1];
    }
    return result;
}

template<typename scalar_t>
at::Tensor
collect_transition_grad(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
) {
    at::Tensor transition_grad = at::zeros({num_labels, num_labels}, aligned_transition_grad.options());
    auto transition_grad_a = transition_grad.accessor<scalar_t, 2>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();
    auto outputs_a = outputs.accessor<int64_t, 2>();
    auto aligned_a = aligned_transition_grad.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < num_batches; ++b) {
        auto cur_output_len = output_lengths_a[b];
        for (int64_t s = 0; s < cur_output_len - 1; ++s) {
            auto cur = outputs_a[b][s];
            auto nxt = outputs_a[b][s + 1];
            transition_grad_a[cur][cur] += aligned_a[0][b][s];
            transition_grad_a[nxt][cur] += aligned_a[1][b][s];
        }
        auto last = outputs_a[cur_output_len - 1];
        transition_grad_a[last][last] += aligned_a[0][b][cur_output_len - 1];
    }
    return transition_grad;
}


template<typename scalar_t>
at::Tensor
collect_input_grad(
        at::Tensor &aligned_input_grad,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_input_len,
        int64_t num_labels
) {
    at::Tensor inputs_grad = at::zeros({batch_input_len, num_batches, num_labels}, aligned_input_grad.options());
    auto inputs_grad_a = inputs_grad.accessor<scalar_t, 3>();
    auto input_lengths_a = input_lengths.accessor<int64_t, 1>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();
    auto outputs_a = outputs.accessor<int64_t, 2>();
    auto aligned_a = aligned_input_grad.accessor<scalar_t, 3>();

#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < num_batches; ++b) {
        for (int64_t t = 0; t < batch_input_len; ++t) {
            if (t < input_lengths_a[b]) {
                for (int64_t s = 0; s < output_lengths_a[b]; ++s) {
                    auto label = outputs[b][s];
                    inputs_grad_a[t][b][label] += aligned_a[t][b][s];
                }
            }
        }
    }

    return inputs_grad_a;
}

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
) {
    at::Tensor aligned_inputs = MY_DISPATCH_FLOAT(make_aligned_inputs,
                                                  inputs, outputs,
                                                  input_lengths, output_lengths,
                                                  batch_input_len,
                                                  num_batches, batch_output_len);

    at::Tensor aligned_transition = MY_DISPATCH_FLOAT(make_aligned_transition,
                                                      transition, outputs,
                                                      input_lengths, output_lengths,
                                                      num_batches, batch_output_len);

    auto alpha_result = force_aligned_alpha_recursion(aligned_inputs, aligned_transition,
                                                      batch_input_len, num_batches, batch_output_len);

    auto alpha = std::get<0>(alpha_result);
    auto path_contrib = std::get<1>(alpha_result);
    auto scores = MY_DISPATCH_FLOAT(collect_scores, alpha, input_lengths, output_lengths, num_batches);

    bool should_roll_inputs = should_roll_to_end(input_lengths, batch_input_len);
    auto aligned_inputs_rolled = should_roll_inputs ? roll_to_end(aligned_inputs, input_lengths) : aligned_inputs;
    auto beta = force_aligned_beta_recursion(aligned_inputs_rolled, aligned_transition,
                                             output_lengths, batch_input_len, num_batches, batch_output_len);
    beta = should_roll_inputs ? roll_to_end(beta, input_lengths, true) : beta;

    return {scores, alpha, beta, path_contrib};
}

void force_aligned_backward(
        at::Tensor &grad_out,
        at::Tensor &alpha,
        at::Tensor &beta,
        at::Tensor &path_contrib
) {

}

}

