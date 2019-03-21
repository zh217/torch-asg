#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <limits>

at::Tensor d_sigmoid(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}


std::vector<at::Tensor>
fac_forward(
        at::Tensor inputs,
        at::Tensor targets,
        at::Tensor transition
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
    auto alpha = at::empty({T, N, S}, torch::CPU(at::kFloat));

    // alpha[0, n, _] <- -inf
    // alpha[0, n, 0] <- inputs[0, n, targets[n, 0]]
    alpha[0] = -std::numeric_limits<float>::infinity();
    for (int n = 0; n != N; ++n) {
        alpha[0][n][0] = 0;
    }

    // FIXME: how do I get rid of the gradfunc non-sense?
    auto trans_next = at::zeros({N, S}, torch::CPU(at::kFloat));
    auto trans_self = at::zeros({N, S}, torch::CPU(at::kFloat));

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
                    s1 = at::logsumexp(at::stack({s1, s2}), 0);
                    std::cout << s1.data<float>()[0] << ' ';
                }
// FIXME should be logadd instead
                alpha[t][n][s] = inputs[t][n][targets[n][s]] + s1;
                std::cout << alpha[t][n][s].data<float>()[0] << '\n';
            }
        }
    }

    return {alpha, trans_next, trans_self};
}


void fac_backward() {

}


void fcc_forward() {

}

void fcc_backward() {

}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fac_forward", &fac_forward, "FAC forward");
}
#endif