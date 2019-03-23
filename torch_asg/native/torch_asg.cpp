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
            const std::string &reduction,
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

        auto scale_fn = get_scale_fn(scale_mode);

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
            const at::Tensor &transition, // num_labels * num_labels
            const at::Tensor &inputs, // batch_input_len * batch_size * num_labels
            const at::Tensor &targets, // batch_size * target_len
            IntArrayRef input_lengths, // batch_size
            IntArrayRef target_lengths, // batch_size
            const std::string &reduction,
            const std::string &scale_mode,
            const at::Tensor &alpha,
            const at::Tensor &scale,
            const at::Tensor &self_trans,
            const at::Tensor &next_trans,
            int64_t batch_input_len,
            int64_t batch_target_len,
            int64_t batch_size,
            int64_t num_labels
    ) {
        at::Tensor beta_cur = at::empty({batch_size, batch_target_len}, alpha.options());
        at::Tensor beta_next = at::empty({batch_size, batch_input_len}, alpha.options());
        at::Tensor grad_transition = at::zeros_like(transition);
        at::Tensor grad_inputs = at::zeros_like(inputs);
        at::Tensor grad_self_trans = at::zeros_like(self_trans);
        at::Tensor grad_next_trans = at::zeros_like(next_trans);

#pragma omp parallel for
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t input_length = std::min(input_lengths[b], batch_input_len);
            int64_t target_length = std::min(target_lengths[b], batch_target_len);

            // beta recursion
            for (int64_t t = input_length - 1; t > 0; --t) {
                int64_t target_frame_lower = t > input_length - target_length ? target_length - (input_length - t) : 0;
                int64_t target_frame_upper = t < target_length ? t + 1 : target_length; // in range(1, target_length)

                // parallel potential
                for (int64_t s = target_frame_lower; s < target_frame_upper; ++s) {

                }
            }

            for (int64_t s = 0; s < target_length; ++s) {

            }
        }

        for (int64_t b = 0; b < batch_size; ++b) {

        }

        return {grad_transition, grad_inputs};
    }


    template<typename scalar_t, at::ScalarType target_scalar_type>
    std::vector<at::Tensor> fcc_loss_cpu_template(
    ) {}

    template<typename scalar_t, at::ScalarType target_scalar_type>
    std::vector<at::Tensor> fcc_loss_backward_cpu_template() {}
}
//
//std::vector<torch::Tensor>
//fac_forward(
//        const torch::Tensor &transition,
//        const torch::Tensor &inputs,
//        const torch::Tensor &targets,
//        const torch::Tensor &input_lengths,
//        const torch::Tensor &target_lengths,
//        const std::string &reduction,
//        const std::string &scale_mode
//) {
//// C: # chars
//// T: input length
//// N: batch size
//// S: target length
//// targets    N * S
//// inputs     T * N * C
//// result     N
//// scale      N
//// alpha      T * N * S
//// transition C * C
//// trans_next N * S
//// trans_self N * S
//    auto T = inputs.size(0);
//    auto N = inputs.size(1);
//    auto C = inputs.size(2);
//    auto S = targets.size(1);
//
//    std::cout << "\nSizes: T " << T << ", N " << N << ", C " << C << ", S " << S << '\n';
//    auto alpha = torch::empty({T, N, S}, torch::TensorOptions().requires_grad(false));
//    alpha.fill_(-std::numeric_limits<float>::infinity());
//
//    auto result = torch::empty({N}, torch::TensorOptions().requires_grad(false));
//
//    // alpha[0, n, _] <- -inf
//    // alpha[0, n, 0] <- inputs[0, n, targets[n, 0]]
//    for (int n = 0; n != N; ++n) {
//        alpha[0][n][0] = 0;
//    }
//
//    auto trans_next = torch::zeros({N, S}, torch::TensorOptions().requires_grad(false));
//    auto trans_self = torch::zeros({N, S}, torch::TensorOptions().requires_grad(false));
//
//    for (int n = 0; n != N; ++n) {
//        auto prev_target = targets[0][0];
//        for (int s = 0; s != S; ++s) {
//            auto target = targets[n][s];
//            trans_self[n][s] = transition[target][target];
//            if (s > 0) {
//                trans_next[n][s] = transition[prev_target][target];
//            }
//            prev_target = target;
//        }
//    }
//
////    s1 <- trans_self[n, s] + alpha[t - 1, n, s]
////    s2 <- trans_next[n, s] + alpha[t - 1, n, s - 1]
////    alpha[t, n, s] <- inputs[t, n, targets[n, s]] + logadd(s1, s2)
//
//    for (int t = 1; t < T; ++t) {
//        for (int n = 0; n != N; ++n) {
//            for (int s = 0; s != S; ++s) {
//                auto s1 = trans_self[n][s] + alpha[t - 1][n][s];
//                std::cout << t << ' ' << n << ' ' << s << ' ' << s1.data<float>()[0] << ' ';
//                if (s > 0) {
//                    auto s2 = trans_next[n][s] + alpha[t - 1][n][s - 1];
//                    std::cout << s2.data<float>()[0] << ' ';
//                    s1 = torch::logsumexp(torch::stack({s1, s2}), 0);
//                    std::cout << s1.data<float>()[0] << ' ';
//                }
//                alpha[t][n][s] = inputs[t][n][targets[n][s]] + s1;
//                std::cout << alpha[t][n][s].data<float>()[0] << '\n';
//            }
//        }
//    }
//
//    for (int n = 0; n != N; ++n) {
//        result[n] = alpha[input_lengths[n] - 1][n][target_lengths[n] - 1];
//    }
//
//    return {alpha, trans_next, trans_self, result};
//}
//
//
//std::vector<torch::Tensor> fac_backward(
//        const torch::Tensor &alpha
//) {
//    auto beta = torch::empty_like(alpha);
//    beta.fill_(-std::numeric_limits<float>::infinity());
//    return {beta};
//}
//
//
//void fcc_forward() {
//
//}
//
//void fcc_backward() {
//
//}
//
//#ifdef TORCH_EXTENSION_NAME
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("fac_forward", &fac_forward, "FAC forward");
//    m.def("fac_backward", &fac_backward, "FAC backward");
//}
//#endif