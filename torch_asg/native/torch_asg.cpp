#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <limits>

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}


std::vector<torch::Tensor>
fac_forward(
        const torch::Tensor &transition,
        const torch::Tensor &inputs,
        const torch::Tensor &targets,
        const torch::Tensor &input_lengths,
        const torch::Tensor &target_lengths,
        const std::string &reduction,
        const std::string &scale_mode
) {
// C: # chars
// T: input length
// N: batch size
// S: target length
// targets    N * S
// inputs     T * N * C
// result     N
// scale      N
// alpha      T * N * S
// transition C * C
// trans_next N * S
// trans_self N * S
    auto T = inputs.size(0);
    auto N = inputs.size(1);
    auto C = inputs.size(2);
    auto S = targets.size(1);

    std::cout << "\nSizes: T " << T << ", N " << N << ", C " << C << ", S " << S << '\n';
    auto alpha = torch::empty({T, N, S}, torch::TensorOptions().requires_grad(false));
    alpha.fill_(-std::numeric_limits<float>::infinity());

    auto result = torch::empty({N}, torch::TensorOptions().requires_grad(false));

    // alpha[0, n, _] <- -inf
    // alpha[0, n, 0] <- inputs[0, n, targets[n, 0]]
    for (int n = 0; n != N; ++n) {
        alpha[0][n][0] = 0;
    }

    auto trans_next = torch::zeros({N, S}, torch::TensorOptions().requires_grad(false));
    auto trans_self = torch::zeros({N, S}, torch::TensorOptions().requires_grad(false));

    for (int n = 0; n != N; ++n) {
        auto prev_target = targets[0][0];
        for (int s = 0; s != S; ++s) {
            auto target = targets[n][s];
            trans_self[n][s] = transition[target][target];
            if (s > 0) {
                trans_next[n][s] = transition[prev_target][target];
            }
            prev_target = target;
        }
    }

//    s1 <- trans_self[n, s] + alpha[t - 1, n, s]
//    s2 <- trans_next[n, s] + alpha[t - 1, n, s - 1]
//    alpha[t, n, s] <- inputs[t, n, targets[n, s]] + logadd(s1, s2)

    for (int t = 1; t < T; ++t) {
        for (int n = 0; n != N; ++n) {
            for (int s = 0; s != S; ++s) {
                auto s1 = trans_self[n][s] + alpha[t - 1][n][s];
                std::cout << t << ' ' << n << ' ' << s << ' ' << s1.data<float>()[0] << ' ';
                if (s > 0) {
                    auto s2 = trans_next[n][s] + alpha[t - 1][n][s - 1];
                    std::cout << s2.data<float>()[0] << ' ';
                    s1 = torch::logsumexp(torch::stack({s1, s2}), 0);
                    std::cout << s1.data<float>()[0] << ' ';
                }
                alpha[t][n][s] = inputs[t][n][targets[n][s]] + s1;
                std::cout << alpha[t][n][s].data<float>()[0] << '\n';
            }
        }
    }

    for (int n = 0; n != N; ++n) {
        result[n] = alpha[input_lengths[n] - 1][n][target_lengths[n] - 1];
    }

    return {alpha, trans_next, trans_self, result};
}


std::vector<torch::Tensor> fac_backward(
        const torch::Tensor &alpha
) {
    auto beta = torch::empty_like(alpha);
    beta.fill_(-std::numeric_limits<float>::infinity());
    return {beta};
}


void fcc_forward() {

}

void fcc_backward() {

}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fac_forward", &fac_forward, "FAC forward");
    m.def("fac_backward", &fac_backward, "FAC backward");
}
#endif