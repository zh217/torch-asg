//
// Created by amade on 4/11/2019.
//

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include "streamlined_fast_gpu.h"
#include "fully_connected_lattice.h"
#include "force_aligned_lattice.h"
#include "force_aligned_lattice_gpu.h"

namespace torch_asg {

inline void _block_waiters(at::cuda::CUDAStream waitee, std::initializer_list<at::cuda::CUDAStream> waiters) {
    at::cuda::CUDAEvent sync{};
    sync.record(waitee);
    for (auto &waiter :waiters) {
        sync.block(waiter);
    }
}

at::Tensor
fast_asg_gpu_forward_only(
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
    // We only need the beta recursions, and the full lattice beta can be calculated concurrently with the aligned
    // lattice beta
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();

    at::Tensor input_lengths_cpu = input_lengths.is_cuda() ? input_lengths.to(at::kCPU, false, true) : input_lengths;
    bool should_roll = should_roll_to_end(input_lengths_cpu, batch_input_len);

    at::cuda::CUDAStream stream1 = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool();

    // pooled stream waits for default stream
    _block_waiters(stream1, {stream2});

    // calculate fully connected beta and collect the score

    auto beta_full = at::full({batch_input_len, num_batches, num_labels}, neg_inf,
                              inputs.options().requires_grad(false));

    at::Tensor input_aligned = should_roll ? roll_to_end(inputs, input_lengths_cpu) : inputs;
    fully_connected_beta_recursion(beta_full, input_aligned, transition, batch_input_len, num_batches, num_labels);
    beta_full = should_roll ? roll_to_end(beta_full, input_lengths_cpu, true) : beta_full;
    auto forward_scores_full = (beta_full[0] + inputs[0]).logsumexp(1);

    at::cuda::setCurrentCUDAStream(stream2);

    // calculate aligned beta and collect the score

    at::Tensor aligned_inputs = MY_DISPATCH_FLOAT(make_aligned_inputs_gpu,
                                                  inputs, outputs,
                                                  input_lengths, output_lengths,
                                                  batch_input_len,
                                                  num_batches, batch_output_len,
                                                  stream2);

    at::Tensor aligned_transition = MY_DISPATCH_FLOAT(make_aligned_transition_gpu,
                                                      transition, outputs,
                                                      input_lengths, output_lengths,
                                                      num_batches, batch_output_len,
                                                      stream2);


    auto aligned_inputs_rolled = should_roll ? roll_to_end(aligned_inputs, input_lengths_cpu) : aligned_inputs;

    at::Tensor beta_aligned = at::full_like(aligned_inputs_rolled, neg_inf); // input_len, batch, output_len

    force_aligned_beta_recursion(beta_aligned, aligned_inputs_rolled, aligned_transition,
                                 output_lengths, batch_input_len, num_batches, batch_output_len);
    beta_aligned = should_roll ? roll_to_end(beta_aligned, input_lengths_cpu, true) : beta_aligned;

    auto forward_scores_aligned = beta_aligned[0].permute({1, 0})[0] + aligned_inputs[0].permute({1, 0})[0];

    // default stream waits for pooled stream
    _block_waiters(stream2, {stream1});

    at::cuda::setCurrentCUDAStream(stream1);

    return forward_scores_full - forward_scores_aligned;

}

std::tuple<
        at::Tensor, // full_scores
        at::Tensor, // aligned_scores
        at::Tensor, // full_gamma
        at::Tensor, // aligned_gamma
        at::Tensor, // full_path_contrib
        at::Tensor  // aligned_path_contrib
>
fast_asg_gpu_forward(
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
    // We use four streams to calculate full-alpha, full-beta, aligned-alpha, aligned-beta concurrently.
    constexpr auto neg_inf = -std::numeric_limits<double>::infinity();

    at::Tensor input_lengths_cpu = input_lengths.is_cuda() ? input_lengths.to(at::kCPU, false, true) : input_lengths;
    bool should_roll = should_roll_to_end(input_lengths_cpu, batch_input_len);

    at::cuda::CUDAStream stream_full_beta = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStream stream_full_alpha = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream stream_aligned_beta = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream stream_aligned_alpha = at::cuda::getStreamFromPool();

    // all pooled streams wait for default stream
    _block_waiters(stream_full_beta, {stream_full_alpha,
                                      stream_aligned_beta,
                                      stream_aligned_alpha});

    // full beta

    at::cuda::setCurrentCUDAStream(stream_full_beta);


    auto beta_full = at::full({batch_input_len, num_batches, num_labels}, neg_inf,
                              inputs.options().requires_grad(false));

    at::Tensor input_aligned = should_roll ? roll_to_end(inputs, input_lengths_cpu) : inputs;
    fully_connected_beta_recursion(beta_full, input_aligned, transition, batch_input_len, num_batches, num_labels);
    beta_full = should_roll ? roll_to_end(beta_full, input_lengths_cpu, true) : beta_full;
    auto forward_scores_full = (beta_full[0] + inputs[0]).logsumexp(1);

    // full alpha

    at::cuda::setCurrentCUDAStream(stream_full_alpha);

    auto alpha_full = at::full({batch_input_len, num_batches, num_labels}, neg_inf,
                               inputs.options().requires_grad(false));
    auto path_contrib_full = at::zeros({batch_input_len - 1, num_batches, num_labels, num_labels},
                                       inputs.options().requires_grad(false));

    fully_connected_alpha_recursion(alpha_full, path_contrib_full, inputs, transition, batch_input_len, num_batches,
                                    num_labels);

    // we are using full_alpha, so we wait for full_beta
    _block_waiters(stream_full_beta, {stream_full_alpha});

    auto gamma_full = beta_full + alpha_full;

    // aligned preparation

    at::Tensor aligned_inputs = MY_DISPATCH_FLOAT(make_aligned_inputs_gpu,
                                                  inputs, outputs,
                                                  input_lengths, output_lengths,
                                                  batch_input_len,
                                                  num_batches, batch_output_len,
                                                  stream_aligned_alpha);

    at::Tensor aligned_transition = MY_DISPATCH_FLOAT(make_aligned_transition_gpu,
                                                      transition, outputs,
                                                      input_lengths, output_lengths,
                                                      num_batches, batch_output_len,
                                                      stream_aligned_beta);

    // let aligned_alpha, aligned_beta wait for each other
    _block_waiters(stream_aligned_beta, {stream_aligned_alpha});
    _block_waiters(stream_aligned_alpha, {stream_aligned_beta});

    // aligned beta

    at::cuda::setCurrentCUDAStream(stream_aligned_beta);

    auto aligned_inputs_rolled = should_roll ? roll_to_end(aligned_inputs, input_lengths_cpu) : aligned_inputs;

    at::Tensor beta_aligned = at::full_like(aligned_inputs_rolled, neg_inf); // input_len, batch, output_len

    force_aligned_beta_recursion(beta_aligned, aligned_inputs_rolled, aligned_transition,
                                 output_lengths, batch_input_len, num_batches, batch_output_len);
    beta_aligned = should_roll ? roll_to_end(beta_aligned, input_lengths_cpu, true) : beta_aligned;

    auto forward_scores_aligned = beta_aligned[0].permute({1, 0})[0] + aligned_inputs[0].permute({1, 0})[0];

    // aligned alpha

    at::cuda::setCurrentCUDAStream(stream_aligned_alpha);

    auto alpha_aligned = aligned_inputs.clone().detach();

    alpha_aligned[0].slice(1, 1).fill_(neg_inf);

    auto path_contrib_aligned = at::zeros({batch_input_len - 1,
                                           2,
                                           num_batches,
                                           batch_output_len - 1}, aligned_inputs.options().requires_grad(false));


    force_aligned_alpha_recursion(alpha_aligned, path_contrib_aligned, aligned_inputs, aligned_transition,
                                  batch_input_len, num_batches, batch_output_len);


    // we are still on alpha, so we wait for beta
    _block_waiters(stream_aligned_beta, {stream_aligned_alpha});

    auto gamma_aligned = alpha_aligned + beta_aligned;

    // finalization

    // go back to default stream
    at::cuda::setCurrentCUDAStream(stream_full_beta);

    // default stream waits for everyone
    _block_waiters(stream_full_alpha, {stream_full_beta});
    _block_waiters(stream_aligned_beta, {stream_full_beta});
    _block_waiters(stream_aligned_alpha, {stream_full_beta});

    return {forward_scores_full, forward_scores_aligned,
            gamma_full, gamma_aligned,
            path_contrib_full, path_contrib_aligned};
}

std::tuple<
        at::Tensor, // grad_transition
        at::Tensor  // grad_inputs
>
fast_asg_gpu_backward(
        at::Tensor &grad_out_full,
        at::Tensor &grad_out_aligned,
        at::Tensor &gamma_full,
        at::Tensor &gamma_aligned,
        at::Tensor &path_contrib_full,
        at::Tensor &path_contrib_aligned,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels,
        int64_t batch_output_len
) {
    // We use two streams to collect inputs and transition grads concurrently.

    at::cuda::CUDAStream stream1 = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool();

    // pooled waits for default
    _block_waiters(stream1, {stream2});

    at::cuda::setCurrentCUDAStream(stream1);

    auto grad_inputs = masked_softmax(gamma_full, 2) * grad_out_full.view({1, num_batches, 1});
    auto grad_transition = (grad_inputs.slice(0, 1).view({batch_input_len - 1, num_batches, num_labels, 1}) *
                            masked_softmax(path_contrib_full, 3)).sum(std::vector<int64_t>({0, 1}));

    at::cuda::setCurrentCUDAStream(stream2);

    auto aligned_grad_results = force_aligned_derivative(grad_out_aligned, gamma_aligned, path_contrib_aligned,
                                                         num_batches, batch_output_len);

    auto aligned_inputs_grad = std::get<0>(aligned_grad_results);
    auto aligned_transition_grad = std::get<1>(aligned_grad_results);

    // two streams synchronize
    _block_waiters(stream1, {stream2});
    _block_waiters(stream2, {stream1});

    MY_DISPATCH_FLOAT(collect_input_grad_gpu,
                      grad_inputs,
                      aligned_inputs_grad, outputs, input_lengths, output_lengths,
                      batch_input_len, num_batches, num_labels,
                      stream1);

    MY_DISPATCH_FLOAT(collect_transition_grad_gpu,
                      grad_transition,
                      aligned_transition_grad, outputs,
                      output_lengths, num_batches, num_labels,
                      stream2);

    // default waits for pooled
    _block_waiters(stream2, {stream1});

    // go back to default
    at::cuda::setCurrentCUDAStream(stream1);

    return std::tuple<at::Tensor, at::Tensor>({grad_transition, grad_inputs});

}

}
