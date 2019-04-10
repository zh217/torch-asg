#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace torch_asg {

template<typename scalar_t>
__global__ void
make_aligned_inputs_kernel(
        scalar_t *__restrict__ aligned,
        const scalar_t *__restrict__ inputs,
        const int64_t *__restrict__ outputs,
        const int64_t *__restrict__ input_lengths,
        const int64_t *__restrict__ output_lengths,

        const int64_t aligned_stride_0,
        const int64_t aligned_stride_1,
        const int64_t aligned_stride_2,

        const int64_t inputs_stride_0,
        const int64_t inputs_stride_1,
        const int64_t inputs_stride_2,

        const int64_t outputs_stride_0,
        const int64_t outputs_stride_1,

        const int64_t input_lengths_stride_0,

        const int64_t output_lengths_stride_0,

        const int64_t batch_input_len,
        const int64_t num_batches,
        const int64_t batch_output_len
) {
    int64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t t = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t s = threadIdx.z + blockIdx.z * blockDim.z;

    if (!(t < batch_input_len && b < num_batches && s < batch_output_len)) {
        return;
    }

    int64_t input_len = input_lengths[b * input_lengths_stride_0];
    int64_t output_len = output_lengths[b * output_lengths_stride_0];

    if (t < input_len && s < output_len) {
        int64_t output_idx = b * outputs_stride_0 + s * outputs_stride_1; // [b][s]
        int64_t aligned_idx = t * aligned_stride_0 + b * aligned_stride_1 + s * aligned_stride_2; // [t][b][s]
        int64_t output_label = outputs[output_idx];
        int64_t inputs_idx = inputs_stride_0 * t + inputs_stride_1 * b +
                             output_label * inputs_stride_2; // inputs_a[t][b][outputs[b][s]]
        aligned[aligned_idx] = inputs[inputs_idx];
    }
}


template<typename scalar_t>
at::Tensor
make_aligned_inputs_gpu(
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

    dim3 block_dim(1, 32, 16); // b, t, s
    dim3 grid_dim(num_batches, (batch_input_len + 31) / 32, (batch_output_len + 15) / 16);
    make_aligned_inputs_kernel<scalar_t>
            <<<grid_dim, block_dim>>>
            (
                    aligned.data<scalar_t>(),
                    inputs.data<scalar_t>(),
                    outputs.data<int64_t>(),
                    input_lengths.data<int64_t>(),
                    output_lengths.data<int64_t>(),

                    aligned.stride(0),
                    aligned.stride(1),
                    aligned.stride(2),

                    inputs.stride(0),
                    inputs.stride(1),
                    inputs.stride(2),

                    outputs.stride(0),
                    outputs.stride(1),

                    input_lengths.stride(0),

                    output_lengths.stride(0),

                    batch_input_len,
                    num_batches,
                    batch_output_len
            );

    return aligned;
}

template
at::Tensor
make_aligned_inputs_gpu<float>(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
);

template
at::Tensor
make_aligned_inputs_gpu<double>(
        at::Tensor &inputs,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t batch_output_len
);


template<typename scalar_t>
__global__ void
make_aligned_transition_kernel(
        scalar_t *__restrict__ aligned_transition,
        const scalar_t *__restrict__ transition,
        const int64_t *__restrict__ outputs,
        const int64_t *__restrict__ output_lengths,

        const int64_t aligned_transition_stride_0,
        const int64_t aligned_transition_stride_1,
        const int64_t aligned_transition_stride_2,

        const int64_t transition_stride_0,
        const int64_t transition_stride_1,

        const int64_t outputs_stride_0,
        const int64_t outputs_stride_1,

        const int64_t output_lengths_stride_0,

        const int64_t num_batches,
        const int64_t batch_output_len
) {
    int64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t s = threadIdx.y + blockIdx.y * blockDim.y;

    if (b >= num_batches || s >= batch_output_len) {
        return;
    }

    int64_t cur_output_len = output_lengths[b * output_lengths_stride_0];

    scalar_t *self_transition = aligned_transition;
    scalar_t *next_transition = aligned_transition + aligned_transition_stride_0;


    if (s < cur_output_len - 1) {
        int64_t cur = outputs[b * outputs_stride_0 + s * outputs_stride_1];
        int64_t nxt = outputs[b * outputs_stride_0 + (s + 1) * outputs_stride_1];
        self_transition[b * aligned_transition_stride_1 + s * aligned_transition_stride_2] =
                transition[cur * transition_stride_0 + cur * transition_stride_1];
        next_transition[b * aligned_transition_stride_1 + s * aligned_transition_stride_2] =
                transition[nxt * transition_stride_0 + cur * transition_stride_1];
    } else if (s == cur_output_len - 1) {
        int64_t last = outputs[b * outputs_stride_0 + s * outputs_stride_1];
        self_transition[b * aligned_transition_stride_1 + s * aligned_transition_stride_2] =
                transition[last * transition_stride_0 + last * transition_stride_1];
    }
}


template<typename scalar_t>
at::Tensor
make_aligned_transition_gpu(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
) {
    at::Tensor aligned = at::zeros({2, num_batches, batch_output_len}, transition.options().requires_grad(false));

    dim3 block_dim(8, 64, 1); // b, s
    dim3 grid_dim((num_batches + 7) / 8, (batch_output_len + 63) / 64, 1);

    make_aligned_transition_kernel<scalar_t>
             <<<grid_dim, block_dim>>>
            (
                    aligned.data<scalar_t>(),
                    transition.data<scalar_t>(),
                    outputs.data<int64_t>(),
                    output_lengths.data<int64_t>(),

                    aligned.stride(0),
                    aligned.stride(1),
                    aligned.stride(2),

                    transition.stride(0),
                    transition.stride(1),

                    outputs.stride(0),
                    outputs.stride(1),

                    output_lengths.stride(0),

                    num_batches,
                    batch_output_len
            );

    return aligned;
}


template
at::Tensor
make_aligned_transition_gpu<float>(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
);

template
at::Tensor
make_aligned_transition_gpu<double>(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
);

template<typename scalar_t>
__global__ void
collect_transition_grad_kernel(
        scalar_t *__restrict__ transition_grad,
        const scalar_t *__restrict__ aligned_transition_grad,
        const int64_t *__restrict__ outputs,
        const int64_t *__restrict__ output_lengths,

        const int64_t transition_grad_stride_0,
        const int64_t transition_grad_stride_1,

        const int64_t aligned_transition_grad_stride_0,
        const int64_t aligned_transition_grad_stride_1,
        const int64_t aligned_transition_grad_stride_2,

        const int64_t outputs_stride_0,
        const int64_t outputs_stride_1,

        const int64_t output_lengths_stride_0,

        const int64_t num_batches,
        const int64_t batch_output_len
) {
    int64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t s = threadIdx.y + blockIdx.y * blockDim.y;

    if (b >= num_batches || s >= batch_output_len) {
        return;
    }

    int64_t cur_output_len = output_lengths[b * output_lengths_stride_0];

    if (s >= cur_output_len) {
        return;
    }

    int64_t cur = outputs[b * outputs_stride_0 + s * outputs_stride_1];

    atomicAdd(transition_grad + cur * transition_grad_stride_0 + cur * transition_grad_stride_1,
              aligned_transition_grad[b * aligned_transition_grad_stride_1 + s * aligned_transition_grad_stride_2]);

    if (s != cur_output_len - 1) {
        int64_t nxt = outputs[b * outputs_stride_0 + (s + 1) * outputs_stride_1];

        atomicAdd(transition_grad + nxt * transition_grad_stride_0 + cur * transition_grad_stride_1,
                  aligned_transition_grad[aligned_transition_grad_stride_0 +
                                          b * aligned_transition_grad_stride_1 +
                                          s * aligned_transition_grad_stride_2]);
    }
}


template<typename scalar_t>
at::Tensor
collect_transition_grad_gpu(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
) {
    at::Tensor transition_grad = at::zeros({num_labels, num_labels},
                                           aligned_transition_grad.options().requires_grad(false));

    int64_t batch_output_len = outputs.size(1);

    dim3 block_dim(8, 64, 1); // b, s
    dim3 grid_dim((num_batches + 7) / 8, (batch_output_len + 63) / 64, 1);

    collect_transition_grad_kernel<scalar_t>
             <<<grid_dim, block_dim>>>
            (
                    transition_grad.data<scalar_t>(),
                    aligned_transition_grad.data<scalar_t>(),
                    outputs.data<int64_t>(),
                    output_lengths.data<int64_t>(),

                    transition_grad.stride(0),
                    transition_grad.stride(1),

                    aligned_transition_grad.stride(0),
                    aligned_transition_grad.stride(1),
                    aligned_transition_grad.stride(2),

                    outputs.stride(0),
                    outputs.stride(1),

                    output_lengths.stride(0),

                    num_batches,
                    batch_output_len
            );

    return transition_grad;
}

template
at::Tensor
collect_transition_grad_gpu<float>(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
);

template
at::Tensor
collect_transition_grad_gpu<double>(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
);

template<typename scalar_t>
__global__ void
collect_input_grad_kernel(
        scalar_t *__restrict__ inputs_grad,
        const scalar_t *__restrict__ aligned_inputs_grad,
        const int64_t *__restrict__ outputs,
        const int64_t *__restrict__ input_lengths,
        const int64_t *__restrict__ output_lengths,

        const int64_t inputs_grad_stride_0,
        const int64_t inputs_grad_stride_1,
        const int64_t inputs_grad_stride_2,

        const int64_t aligned_inputs_grad_stride_0,
        const int64_t aligned_inputs_grad_stride_1,
        const int64_t aligned_inputs_grad_stride_2,


        const int64_t outputs_stride_0,
        const int64_t outputs_stride_1,

        const int64_t input_lengths_stride_0,

        const int64_t output_lengths_stride_0,

        const int64_t batch_input_len,
        const int64_t num_batches,
        const int64_t batch_output_len
) {
    int64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t t = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t s = threadIdx.z + blockIdx.z * blockDim.z;

    if (b >= num_batches || t >= batch_input_len || s >= batch_output_len) {
        return;
    }

    int64_t input_len = input_lengths[b * input_lengths_stride_0];

    if (t >= input_len) {
        return;
    }

    int64_t output_len = output_lengths[b * output_lengths_stride_0];

    if (s >= output_len) {
        return;
    }

    int64_t label = outputs[b * outputs_stride_0 + s * outputs_stride_1];
    atomicAdd(inputs_grad + t * inputs_grad_stride_0 + b * inputs_grad_stride_1 + label * inputs_grad_stride_2,
              aligned_inputs_grad[t * aligned_inputs_grad_stride_0 + b * aligned_inputs_grad_stride_1 +
                                  s * aligned_inputs_grad_stride_2]);

}


template<typename scalar_t>
at::Tensor
collect_input_grad_gpu(
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

    int64_t batch_output_len = outputs.size(1);

    dim3 block_dim(1, 32, 16); // b, t, s
    dim3 grid_dim(num_batches, (batch_input_len + 31) / 32, (batch_output_len + 15) / 16);

    collect_input_grad_kernel<scalar_t>
             <<<grid_dim, block_dim>>>
            (
                    inputs_grad.data<scalar_t>(),
                    aligned_input_grad.data<scalar_t>(),
                    outputs.data<int64_t>(),
                    input_lengths.data<int64_t>(),
                    output_lengths.data<int64_t>(),

                    inputs_grad.stride(0),
                    inputs_grad.stride(1),
                    inputs_grad.stride(2),

                    aligned_input_grad.stride(0),
                    aligned_input_grad.stride(1),
                    aligned_input_grad.stride(2),

                    outputs.stride(0),
                    outputs.stride(1),

                    input_lengths.stride(0),

                    output_lengths.stride(0),

                    batch_input_len,
                    num_batches,
                    batch_output_len
            );

    return inputs_grad;
}

template
at::Tensor
collect_input_grad_gpu<float>(
        at::Tensor &aligned_input_grad,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
);

template
at::Tensor
collect_input_grad_gpu<double>(
        at::Tensor &aligned_input_grad,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t batch_input_len,
        int64_t num_batches,
        int64_t num_labels
);

}
