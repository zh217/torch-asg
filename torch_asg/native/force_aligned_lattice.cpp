//
// Created by amade on 4/2/2019.
//

#include "force_aligned_lattice.h"
#include "force_aligned_lattice_gpu.h"

#include <omp.h>
#include <limits>

#define MY_DISPATCH_FLOAT_AND_DEVICE(func, fst_arg, ...) \
(fst_arg.dtype() == at::kFloat) ? \
(fst_arg.is_cuda() ? func ## _gpu<float>(fst_arg, ##__VA_ARGS__) : func ## _cpu<float>(fst_arg, ##__VA_ARGS__)) : \
(fst_arg.is_cuda() ? func ## _gpu<double>(fst_arg, ##__VA_ARGS__) : func ## _cpu<double>(fst_arg, ##__VA_ARGS__))

namespace torch_asg {


template<typename scalar_t>
at::Tensor
make_aligned_inputs_cpu(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
) {
    constexpr auto neg_inf = -std::numeric_limits<scalar_t>::infinity();
    at::Tensor aligned = at::full({batch_input_len, num_batches, batch_output_len}, neg_inf,
                                  inputs.options().requires_grad(false));
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
make_aligned_transition_cpu(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
) {
//    constexpr auto neg_inf = -std::numeric_limits<scalar_t>::infinity();
    at::Tensor aligned = at::zeros({2, num_batches, batch_output_len}, transition.options().requires_grad(false));
    auto transition_a = transition.accessor<scalar_t, 2>();
    auto aligned_a = aligned.accessor<scalar_t, 3>();
    auto outputs_a = outputs.accessor<int64_t, 2>();
    auto output_lengths_a = output_lengths.accessor<int64_t, 1>();

#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < num_batches; ++b) {
        for (int64_t s = 0; s < batch_output_len; ++s) {
            auto cur_output_len = output_lengths_a[b];
            if (s < cur_output_len - 1) {
                auto cur = outputs_a[b][s];
                auto nxt = outputs_a[b][s + 1];
                aligned_a[0][b][s] = transition_a[cur][cur];
                aligned_a[1][b][s] = transition_a[nxt][cur];
            } else if (s == cur_output_len - 1) {
                auto last = outputs_a[b][cur_output_len - 1];
                aligned_a[0][b][cur_output_len - 1] = transition_a[last][last];
            }
        }
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
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();

    auto alpha = aligned_inputs.clone().detach();

    alpha[0].slice(1, 1).fill_(neg_inf);

    auto alpha_inv_idx = alpha.permute({2, 1, 0}); // output_len, num_batches, input_len
    auto aligned_inputs_inv_idx = aligned_inputs.permute({2, 1, 0});
    auto path_contrib = at::zeros({batch_input_len - 1,
                                   2,
                                   num_batches,
                                   batch_output_len - 1}, aligned_inputs.options().requires_grad(false));
    auto self_transition = aligned_transition[0]; // num_batches, batch_output_len
    auto next_transition = aligned_transition[1]; // num_batches, batch_output_len

    alpha_inv_idx[0].slice(1, 1) += self_transition.permute({1, 0})[0].view({num_batches, 1});

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
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();

    at::Tensor beta = at::full_like(aligned_inputs, neg_inf); // input_len, batch, output_len

    auto self_transition = aligned_transition[0]; // batch, output_len
    auto next_transition = aligned_transition[1]; // batch, output_len

    for (int64_t b = 0; b < num_batches; ++b) {
        beta[batch_input_len - 1][b][output_lengths[b] - 1] = 0;
    }

    auto beta_last_row = aligned_inputs.permute({2, 0, 1})[batch_output_len - 1].slice(0, 1, batch_input_len)
                         // ^^ input_len, batch, remove input_len = 0
                         // vv output_len, batch -> batch -> 1, batch
                         + self_transition.permute({1, 0})[batch_output_len - 1].view({1, num_batches});

    beta_last_row = beta_last_row.flip(0);
    beta_last_row = beta_last_row.cumsum(0);
    beta_last_row = beta_last_row.flip(0);

    // output_len, input_len, batch -> input_len, batch -> remove last idx
    beta.permute({2, 0, 1})[batch_output_len - 1].slice(0, 0, batch_input_len - 1) = beta_last_row;

    auto beta_no_top = beta.slice(2, 1, batch_output_len);
    auto beta_no_bottom = beta.slice(2, 0, batch_output_len - 1);

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

    return beta;
}

std::tuple<at::Tensor, at::Tensor>
force_aligned_derivative(
        at::Tensor &grad_out, // num_batches
        at::Tensor &gamma, // batch_input_len, num_batches, batch_output_len
        at::Tensor &path_contrib, // batch_input_len - 1, 2, num_batches, batch_output_len - 1
        int64_t num_batches,
        int64_t batch_output_len
) {
    auto aligned_inputs_grad = masked_softmax(gamma, 2) * grad_out.view({1, num_batches, 1});
    auto path_factor = masked_softmax(path_contrib, 1).permute(
            {1, 0, 2, 3}); // <<2>>, batch_input_len - 1, num_batches, batch_output_len - 1
    auto hori_factor = path_factor[0]; // batch_input_len - 1, num_batches, batch_output_len - 1
    auto diag_factor = path_factor[1]; // batch_input_len - 1, num_batches, batch_output_len - 1

    at::Tensor aligned_transition_grad = at::zeros({2, num_batches, batch_output_len},
                                                   gamma.options().requires_grad(false));
    auto self_trans_grad = aligned_transition_grad[0];
    auto next_trans_grad = aligned_transition_grad[1];


    self_trans_grad.permute({1, 0})[0] = aligned_inputs_grad.permute({2, 0, 1})[0].slice(0, 1).sum(0);

    auto state_factor = aligned_inputs_grad.slice(0, 1).slice(2, 1);

    self_trans_grad.slice(1, 1) = (state_factor * hori_factor).sum(0);
    next_trans_grad.slice(1, 0, batch_output_len - 1) = (state_factor * diag_factor).sum(0);

    return {aligned_inputs_grad, aligned_transition_grad};
}

template<typename scalar_t>
at::Tensor
collect_scores(
        at::Tensor &alpha,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches
) {
    at::Tensor result = at::zeros({num_batches}, alpha.options().requires_grad(false));
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
collect_transition_grad_cpu(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
) {
    at::Tensor transition_grad = at::zeros({num_labels, num_labels},
                                           aligned_transition_grad.options().requires_grad(false));
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
        auto last = outputs_a[b][cur_output_len - 1];
        transition_grad_a[last][last] += aligned_a[0][b][cur_output_len - 1];
    }
    return transition_grad;
}


template<typename scalar_t>
at::Tensor
collect_input_grad_cpu(
        at::Tensor &aligned_input_grad,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
) {
    at::Tensor inputs_grad = at::zeros({batch_input_len, num_batches, num_labels},
                                       aligned_input_grad.options().requires_grad(false));
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
                    auto label = outputs_a[b][s];
                    inputs_grad_a[t][b][label] += aligned_a[t][b][s];
                }
            }
        }
    }

    return inputs_grad;
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
    at::Tensor input_lengths_cpu = input_lengths.is_cuda() ? input_lengths.to(at::kCPU, false, true) : input_lengths;
    at::Tensor aligned_inputs = MY_DISPATCH_FLOAT_AND_DEVICE(make_aligned_inputs,
                                                             inputs, outputs,
                                                             input_lengths, output_lengths,
                                                             batch_input_len,
                                                             num_batches, batch_output_len);

    at::Tensor aligned_transition = MY_DISPATCH_FLOAT_AND_DEVICE(make_aligned_transition,
                                                                 transition, outputs,
                                                                 input_lengths, output_lengths,
                                                                 num_batches, batch_output_len);

    auto alpha_result = force_aligned_alpha_recursion(aligned_inputs, aligned_transition,
                                                      batch_input_len, num_batches, batch_output_len);

    auto alpha = std::get<0>(alpha_result);
    auto path_contrib = std::get<1>(alpha_result);

    bool should_roll_inputs = should_roll_to_end(input_lengths_cpu, batch_input_len);
    auto aligned_inputs_rolled = should_roll_inputs ? roll_to_end(aligned_inputs, input_lengths_cpu) : aligned_inputs;
    auto beta = force_aligned_beta_recursion(aligned_inputs_rolled, aligned_transition,
                                             output_lengths, batch_input_len, num_batches, batch_output_len);
    beta = should_roll_inputs ? roll_to_end(beta, input_lengths_cpu, true) : beta;

//    auto scores = MY_DISPATCH_FLOAT(collect_scores, alpha, input_lengths, output_lengths, num_batches);
    auto scores = beta[0].permute({1, 0})[0] + aligned_inputs[0].permute({1, 0})[0];

    return {scores, alpha, beta, path_contrib};
}

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
) {
    auto gamma = alpha + beta;
    auto grad_results = force_aligned_derivative(grad_out, gamma, path_contrib, num_batches, batch_output_len);
    auto aligned_inputs_grad = std::get<0>(grad_results);
    auto aligned_transition_grad = std::get<1>(grad_results);

    auto inputs_grad = MY_DISPATCH_FLOAT_AND_DEVICE(collect_input_grad,
                                                    aligned_inputs_grad, outputs, input_lengths, output_lengths,
                                                    batch_input_len, num_batches, num_labels);
    auto transition_grad = MY_DISPATCH_FLOAT_AND_DEVICE(collect_transition_grad,
                                                        aligned_transition_grad, outputs,
                                                        output_lengths, num_batches, num_labels);
    return {transition_grad, inputs_grad};
}

}

