//#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

#include <numeric>
#include <type_traits>

#include <iostream>
#include <vector>
#include <limits>

namespace torch_asg {

using IntArrayRef = at::ArrayRef<int64_t>;

using CriterionScaleFn = std::function<float(int64_t /* alphabet size */, int64_t /* timeframes */,
                                             int64_t /* labelsize */)>;

template<typename scalar_t>
inline scalar_t _log_sum_exp(scalar_t log_a, scalar_t log_b) {
    constexpr scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();

    if (log_a < log_b) {
        std::swap(log_a, log_b);
    }
    if (log_b == neg_inf) {
        return log_a;
    }
    return log_a + std::log1p(std::exp(log_b - log_a));
}

template<typename scalar_t>
inline void _d_log_sum_exp_2(scalar_t a, scalar_t b, scalar_t &grad_a, scalar_t &grad_b) {
    scalar_t m = std::max(a, b);
    a = std::exp(a - m);
    b = std::exp(b - m);
    scalar_t z = a + b;
    grad_a = a / z;
    grad_b = a / z;
}

CriterionScaleFn get_scale_fn(const std::string &scale_mode) {
    if (scale_mode == "none") {
        return [](int64_t /* unused */, int64_t /* unused */, int64_t /* unused */) {
            return 1.0;
        };
    } else if (scale_mode == "input_size") {
        return [](int64_t /* unused */, int64_t T, int64_t /* unused */) {
            return (T > 0) ? (1.0 / T) : 1.0;
        };
    } else if (scale_mode == "input_size_sqrt") {
        return [](int64_t /* unused */, int64_t T, int64_t /* unused */) {
            return (T > 0) ? std::sqrt(1.0 / T) : 1.0;
        };
    } else if (scale_mode == "target_size") {
        return [](int64_t /* unused */, int64_t /* unused */, int64_t L) {
            return (L > 0) ? (1.0 / L) : 1.0;
        };
    } else if (scale_mode == "target_size_sqrt") {
        return [](int64_t /* unused */, int64_t /* unused */, int64_t L) {
            return (L > 0) ? std::sqrt(1.0 / L) : 1.0;
        };
    } else {
        throw std::runtime_error("Unknown scale_mode: " + scale_mode);
    }
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fac_loss_cpu_template(
        const at::Tensor &transition, // num_labels * num_labels
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const std::string &scale_mode
) {
    // constants
    constexpr scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
    using target_t = typename std::conditional<target_scalar_type == at::kInt, int, int64_t>::type;

    // sanity check
    at::CheckedFrom c = "fac_loss_cpu";
    auto transition_arg = at::TensorArg(transition, "transition", 1);
    auto inputs_arg = at::TensorArg(inputs, "inputs", 2);
    auto targets_arg = at::TensorArg(targets, "targets", 3);

    at::checkScalarType(c, targets_arg, target_scalar_type);
    at::checkDim(c, transition_arg, 2);
    at::checkDim(c, inputs_arg, 3);
    at::checkDim(c, targets_arg, 2);

    int64_t batch_input_len = inputs.size(0);
    int64_t batch_target_len = targets.size(1);
    int64_t batch_size = inputs.size(1);
    int64_t num_labels = inputs.size(2);

    AT_CHECK(transition.size(0) == num_labels && transition.size(1) == num_labels,
             "inputs/transition matrix size mismatch");
    AT_CHECK(targets.size(0) == batch_size, "inputs/targets batch size mismatch");
    AT_CHECK(input_lengths.size() == batch_size, "input/input_lengths batch size mismatch");
    AT_CHECK(target_lengths.size() == batch_size, "input/target_lengths batch size mismatch");

    auto scale_fn = get_scale_fn(scale_mode);

    at::Tensor out = at::empty({batch_size}, inputs.options());
    at::Tensor scale = at::empty({batch_size}, inputs.options());
    at::Tensor alpha = at::empty({batch_size, batch_input_len, batch_target_len}, inputs.options());
    at::Tensor self_trans = at::empty({batch_size, batch_target_len}, inputs.options());
    at::Tensor next_trans = at::empty({batch_size, batch_target_len}, inputs.options());

    auto inputs_bf = inputs.permute({1, 0, 2}); // bf for batch-first

    auto transition_a = transition.accessor<scalar_t>(2);
    auto inputs_bf_a = inputs_bf.accessor<scalar_t>(3);
    auto targets_a = targets.accessor<target_t>(2);
    auto alpha_a = alpha.accessor<scalar_t>(3);
    auto self_trans_a = self_trans.accessor<scalar_t>(2);
    auto next_trans_a = next_trans.accessor<scalar_t>(2);

#pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t input_length = std::min(input_lengths[b], batch_input_len);
        int64_t target_length = std::min(target_lengths[b], batch_target_len);
        auto inputs_cur_batch_a = inputs_bf_a[b]; // batch_input_len * num_labels
        auto targets_cur_batch_a = targets_a[b]; // batch_target_len
        auto alpha_cur_batch_a = alpha_a[b]; // batch_input_len * batch_target_len
        auto self_trans_cur_batch_a = self_trans_a[b]; // batch_target_len
        auto next_trans_cur_batch_a = next_trans_a[b]; // batch_target_len

        target_length = std::min(input_length, target_length);
        AT_CHECK(target_length > 0, "Target size cannot be 0");

        alpha_cur_batch_a[0][0] = inputs_cur_batch_a[0][targets_cur_batch_a[0]];

        target_t last_target = targets_cur_batch_a[0];
        self_trans_cur_batch_a[0] = transition_a[last_target][last_target];
        next_trans_cur_batch_a[0] = 0.;

        for (int64_t s = 1; s < target_length; ++s) {
            target_t cur_target = targets_cur_batch_a[s];
            self_trans_cur_batch_a[s] = transition_a[cur_target][cur_target];
            next_trans_cur_batch_a[s] = transition_a[cur_target][last_target];
        }

        auto alpha_prev_frame_a = alpha_cur_batch_a[0];

        for (int64_t t = 1; t < input_length; ++t) {
            auto alpha_cur_frame_a = alpha_cur_batch_a[t];
            auto inputs_cur_frame_a = inputs_cur_batch_a[t];
            int64_t target_frame_lower = t > input_length - target_length ? target_length - (input_length - t) : 1;
            int64_t target_frame_upper = std::min(t, target_length); // in range(1, target_length)

            if (t <= input_length - target_length) {
                // still at the top row
                alpha_cur_batch_a[0] =
                        self_trans_cur_batch_a[0] +
                        alpha_prev_frame_a[0] +
                        inputs_cur_frame_a[targets_cur_batch_a[0]];
            }

            // parallel potential
            for (int64_t s = target_frame_lower; s < target_frame_upper; ++s) {
                scalar_t hori_route = self_trans_cur_batch_a[s] + alpha_prev_frame_a[s];
                scalar_t diag_route = next_trans_cur_batch_a[s] + alpha_prev_frame_a[s - 1];
                alpha_cur_frame_a[s] = _log_sum_exp(hori_route, diag_route) + inputs_cur_frame_a[s];
            }

            if (target_frame_upper < target_length) {
                alpha_cur_frame_a[target_frame_upper] =
                        next_trans_cur_batch_a[target_frame_upper] +
                        alpha_prev_frame_a[target_frame_upper - 1] +
                        inputs_cur_frame_a[targets_cur_batch_a[target_frame_upper]];
            }

            alpha_prev_frame_a = alpha_cur_frame_a;
        }

        scale[b] = scale_fn(num_labels, input_length, target_length);
        out[b] = alpha_cur_batch_a[input_length - 1][target_length - 1];
    }


    return {out, alpha, scale, self_trans, next_trans};
}


template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fac_loss_backward_cpu_template(
        const at::Tensor &grad_out,
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const at::Tensor &alpha,
        const at::Tensor &scale,
        const at::Tensor &self_trans,
        const at::Tensor &next_trans,
        int64_t batch_input_len,
        int64_t batch_target_len,
        int64_t batch_size,
        int64_t num_labels
) {

    using target_t = typename std::conditional<target_scalar_type == at::kInt, int, int64_t>::type;

    at::Tensor beta_cur = at::empty({batch_size, batch_target_len}, alpha.options());
    at::Tensor beta_prev = at::empty({batch_size, batch_input_len}, alpha.options());
    at::Tensor grad_transition = at::zeros({num_labels, num_labels}, alpha.options());
    at::Tensor grad_inputs = at::zeros_like(inputs);
    at::Tensor grad_self_trans = at::zeros_like(self_trans);
    at::Tensor grad_next_trans = at::zeros_like(next_trans);

    auto grad_inputs_bf = grad_inputs.permute({1, 0, 2});

    auto beta_cur_a = beta_cur.accessor<scalar_t>(2);
    auto beta_prev_a = beta_prev.accessor<scalar_t>(2);
    auto grad_transition_a = grad_transition.accessor<scalar_t>(2);
    auto grad_inputs_bf_a = grad_inputs_bf.accessor<scalar_t>(3);
    auto grad_self_trans_a = grad_self_trans.accessor<scalar_t>(2);
    auto grad_next_trans_a = grad_next_trans.accessor<scalar_t>(2);
    auto self_trans_a = self_trans.accessor<scalar_t>(2);
    auto next_trans_a = next_trans.accessor<scalar_t>(2);

    auto inputs_bf = inputs.permute({1, 0, 2}); // bf for batch-first

    auto alpha_a = alpha.accessor<scalar_t>(3);
    auto targets_a = targets.accessor<scalar_t>(2);

#pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const scalar_t grad_batch = scale[b] * grad_out[b];
        int64_t input_length = std::min(input_lengths[b], batch_input_len);
        int64_t target_length = std::min(target_lengths[b], batch_target_len);
        auto grad_input_cur_batch_a = grad_inputs_bf_a[b];
        auto alpha_cur_batch_a = alpha_a[b];
        auto targets_cur_batch_a = targets_a[b];
        auto grad_self_trans_batch_a = grad_self_trans_a[b];
        auto grad_next_trans_batch_a = grad_next_trans_a[b];
        auto self_trans_cur_batch_a = self_trans_a[b]; // batch_target_len
        auto next_trans_cur_batch_a = next_trans_a[b]; // batch_target_len

        beta_cur_a[target_length - 1] = 1.;

        // beta recursion
        for (int64_t t = input_length - 1; t > 0; --t) {
            auto grad_input_cur_frame_a = grad_input_cur_batch_a[t];
            auto alpha_prev_frame_a = alpha_cur_batch_a[t - 1];

            beta_prev.zero_();

            int64_t target_frame_lower = t > input_length - target_length ? target_length - (input_length - t) : 0;
            int64_t target_frame_upper = t < target_length ? t + 1 : target_length; // in range(1, target_length)

            // parallel potential
            for (int64_t s = target_frame_lower; s < target_frame_upper; ++s) {
                scalar_t cur_elem_grad = beta_cur_a[s];

                grad_input_cur_frame_a[targets_cur_batch_a[s]] += grad_batch * cur_elem_grad;

                if ((target_frame_upper < target_length || s == target_length - 1) &&
                    s == target_frame_upper - 1 &&
                    s > 0) {
                    // left wedge
                    beta_prev_a[s - 1] += beta_cur_a[s];
                    grad_next_trans_batch_a[s] += cur_elem_grad;
                } else if (s == 0) {
                    // on the top row
                    beta_prev_a[s] += beta_cur_a[s];
                    grad_self_trans_batch_a[s] += cur_elem_grad;
                } else {
                    // general case: need to merge contribution from two paths
                    scalar_t hori_route = alpha_prev_frame_a[s] + self_trans_cur_batch_a[s];
                    scalar_t diag_route = alpha_prev_frame_a[s - 1] + next_trans_cur_batch_a[s];
                    scalar_t grad_hori_route, grad_diag_route;
                    _d_log_sum_exp_2(hori_route, diag_route, grad_hori_route, grad_diag_route);

                    grad_hori_route *= cur_elem_grad;
                    grad_diag_route *= cur_elem_grad;

                    grad_self_trans_batch_a[s] += grad_hori_route;
                    grad_next_trans_batch_a[s] += grad_diag_route;

                    beta_prev_a[s] += grad_hori_route;
                    beta_prev_a[s - 1] += grad_diag_route;
                }
            }

            std::swap(beta_cur_a, beta_prev_a);
            std::swap(beta_cur, beta_prev);
        }
    }

    for (int64_t b = 0; b < batch_size; ++b) {
        scalar_t batch_grad = scale[b] * grad_out[b];
        target_t prev_target = targets_a[0];

        grad_transition_a[prev_target][prev_target] += grad_self_trans_a[0] * batch_grad;

        for (int64_t s = 1; s < target_lengths[b]; ++s) {
            target_t cur_target = targets_a[s];
            grad_transition_a[cur_target][cur_target] += grad_self_trans_a[s] * batch_grad;
            grad_transition_a[cur_target][prev_target] += grad_next_trans_a[s] * batch_grad;
            prev_target = cur_target;
        }
    }

    return {grad_transition, grad_inputs};
}


template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fcc_loss_cpu_template(
        const at::Tensor &transition, // num_labels * num_labels
        const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
        const at::Tensor &targets, // batch_size * target_len
        IntArrayRef input_lengths, // batch_size
        IntArrayRef target_lengths, // batch_size
        const std::string &scale_mode
) {
    // constants
    constexpr scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
    using target_t = typename std::conditional<target_scalar_type == at::kInt, int, int64_t>::type;

    // sanity check
    at::CheckedFrom c = "fcc_loss_cpu";
    auto transition_arg = at::TensorArg(transition, "transition", 1);
    auto inputs_arg = at::TensorArg(inputs, "inputs", 2);
    auto targets_arg = at::TensorArg(targets, "targets", 3);

    at::checkScalarType(c, targets_arg, target_scalar_type);
    at::checkDim(c, transition_arg, 2);
    at::checkDim(c, inputs_arg, 3);
    at::checkDim(c, targets_arg, 2);

    int64_t batch_input_len = inputs.size(0);
    int64_t batch_target_len = targets.size(1);
    int64_t batch_size = inputs.size(1);
    int64_t num_labels = inputs.size(2);

    AT_CHECK(transition.size(0) == num_labels && transition.size(1) == num_labels,
             "inputs/transition matrix size mismatch");
    AT_CHECK(targets.size(0) == batch_size, "inputs/targets batch size mismatch");
    AT_CHECK(input_lengths.size() == batch_size, "input/input_lengths batch size mismatch");
    AT_CHECK(target_lengths.size() == batch_size, "input/target_lengths batch size mismatch");

    auto scale_fn = get_scale_fn(scale_mode);

    at::Tensor out = at::empty({batch_size}, inputs.options());
    at::Tensor scale = at::empty({batch_size}, inputs.options());
    at::Tensor alpha = at::empty({batch_size, batch_input_len, num_labels}, inputs.options());
    at::Tensor alpha_max_idx = at::empty({batch_size, batch_input_len}, targets.options());

    auto inputs_bf = inputs.permute({1, 0, 2}); // bf for batch-first

    auto transition_a = transition.accessor<scalar_t>(2);
    auto inputs_bf_a = inputs_bf.accessor<scalar_t>(3);
    auto targets_a = targets.accessor<target_t>(2);
    auto alpha_a = alpha.accessor<scalar_t>(3);

#pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
    }

    return {};
}

template<typename scalar_t, at::ScalarType target_scalar_type>
std::vector<at::Tensor> fcc_loss_backward_cpu_template() {
    return {};
}

}
//#ifdef TORCH_EXTENSION_NAME
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("fac_forward", &fac_forward, "FAC forward");
//    m.def("fac_backward", &fac_backward, "FAC backward");
//}
//#endif