//
// Created by amade on 4/11/2019.
//

#include <ATen/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include "streamlined_fast_gpu.h"
#include "fully_connected_lattice.h"
#include "force_aligned_lattice.h"
#include "force_aligned_lattice_gpu.h"

namespace torch_asg {

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

    at::cuda::CUDAEvent first_sync{};

    first_sync.record(stream1);
    first_sync.block(stream2);

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

    at::cuda::CUDAEvent second_sync{};

    second_sync.record(stream2);
    second_sync.block(stream1);

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

    at::cuda::CUDAEvent init_sync{};

    init_sync.record(stream_full_beta);

    init_sync.block(stream_full_alpha);
    init_sync.block(stream_aligned_beta);
    init_sync.block(stream_aligned_alpha);

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

    at::cuda::CUDAEvent full_final_sync{};
    full_final_sync.record(stream_full_beta);
    full_final_sync.block(stream_full_alpha);

    auto gamma_full = beta_full + alpha_full;

    // aligned preparation

    at::cuda::setCurrentCUDAStream(stream_aligned_beta);

    at::Tensor aligned_inputs = MY_DISPATCH_FLOAT(make_aligned_inputs_gpu,
                                                  inputs, outputs,
                                                  input_lengths, output_lengths,
                                                  batch_input_len,
                                                  num_batches, batch_output_len,
                                                  stream_aligned_beta);

    at::Tensor aligned_transition = MY_DISPATCH_FLOAT(make_aligned_transition_gpu,
                                                      transition, outputs,
                                                      input_lengths, output_lengths,
                                                      num_batches, batch_output_len,
                                                      stream_aligned_beta);

    at::cuda::CUDAEvent aligned_sync{};
    aligned_sync.record(stream_aligned_beta);
    aligned_sync.block(stream_aligned_alpha);

    // aligned beta

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


    at::cuda::CUDAEvent aligned_final_sync{};
    aligned_final_sync.record(stream_aligned_beta);
    aligned_final_sync.block(stream_aligned_alpha);

    auto gamma_aligned = alpha_aligned + beta_aligned;

    // finalization

    at::cuda::setCurrentCUDAStream(stream_full_beta);

    at::cuda::CUDAEvent final_sync_f_a{};
    at::cuda::CUDAEvent final_sync_a_b{};
    at::cuda::CUDAEvent final_sync_a_a{};

    final_sync_f_a.record(stream_full_alpha);
    final_sync_a_b.record(stream_aligned_beta);
    final_sync_a_a.record(stream_aligned_alpha);

    final_sync_f_a.block(stream_full_beta);
    final_sync_a_b.block(stream_full_beta);
    final_sync_a_a.block(stream_full_beta);

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

    at::cuda::CUDAEvent init_sync{};
    init_sync.record(stream1);
    init_sync.block(stream2);

    at::cuda::setCurrentCUDAStream(stream1);

    auto grad_inputs = masked_softmax(gamma_full, 2) * grad_out_full.view({1, num_batches, 1});
    auto grad_transition = (grad_inputs.slice(0, 1).view({batch_input_len - 1, num_batches, num_labels, 1}) *
                            masked_softmax(path_contrib_full, 3)).sum({0, 1});

    at::cuda::setCurrentCUDAStream(stream2);

    auto aligned_grad_results = force_aligned_derivative(grad_out_aligned, gamma_aligned, path_contrib_aligned,
                                                         num_batches, batch_output_len);

    auto aligned_inputs_grad = std::get<0>(aligned_grad_results);
    auto aligned_transition_grad = std::get<1>(aligned_grad_results);

    at::cuda::CUDAEvent mid_sync_1{};
    mid_sync_1.record(stream1);
    mid_sync_1.block(stream2);

    at::cuda::CUDAEvent mid_sync_2{};
    mid_sync_2.record(stream2);
    mid_sync_2.block(stream1);

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

    at::cuda::CUDAEvent final_sync{};
    final_sync.record(stream2);
    final_sync.block(stream1);

    at::cuda::setCurrentCUDAStream(stream1);

    return {grad_transition, grad_inputs};

}

}