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

template
__global__ void
make_aligned_inputs_kernel<float>(
        float *__restrict__ aligned,
        const float *__restrict__ inputs,
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
);

template
__global__ void
make_aligned_inputs_kernel<double>(
        double *__restrict__ aligned,
        const double *__restrict__ inputs,
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
                transition[cur * transition_stride_0 + nxt * transition_stride_1];
    } else if (s == cur_output_len - 1) {
        int64_t last = outputs[b * outputs_stride_0 + s * outputs_stride_1];
        self_transition[b * aligned_transition_stride_1 + s * aligned_transition_stride_2] =
                transition[last * transition_stride_0 + last * transition_stride_1];
    }
}

template
__global__ void
make_aligned_transition_kernel<float>(
        float *__restrict__ aligned_transition,
        const float *__restrict__ transition,
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
);


template
__global__ void
make_aligned_transition_kernel<double>(
        double *__restrict__ aligned_transition,
        const double *__restrict__ transition,
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
//     for (int64_t b = 0; b < num_batches; ++b) {
//        auto cur_output_len = output_lengths_a[b];
//        for (int64_t s = 0; s < cur_output_len - 1; ++s) {
//            auto cur = outputs_a[b][s];
//            auto nxt = outputs_a[b][s + 1];
//            transition_grad_a[cur][cur] += aligned_a[0][b][s];
//            transition_grad_a[nxt][cur] += aligned_a[1][b][s];
//        }
//        auto last = outputs_a[b][cur_output_len - 1];
//        transition_grad_a[last][last] += aligned_a[0][b][cur_output_len - 1];
//    }
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


template
__global__ void
collect_transition_grad_kernel<float>(
        float *__restrict__ transition_grad,
        const float *__restrict__ aligned_transition_grad,
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
);


template
__global__ void
collect_transition_grad_kernel<double>(
        double *__restrict__ transition_grad,
        const double *__restrict__ aligned_transition_grad,
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

    int64_t output_len = output_lengths[s * output_lengths_stride_0];

    if (s >= output_len) {
        return;
    }

    int64_t label = outputs[b * outputs_stride_0 + s * outputs_stride_1];
    atomicAdd(inputs_grad + t * inputs_grad_stride_0 + b * inputs_grad_stride_1 + s * label,
              aligned_inputs_grad[t * aligned_inputs_grad_stride_0 + b * aligned_inputs_grad_stride_1 +
                                  s * aligned_inputs_grad_stride_2]);

}

template
__global__ void
collect_input_grad_kernel<float>(
        float *__restrict__ inputs_grad,
        const float *__restrict__ aligned_inputs_grad,
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
);


template
__global__ void
collect_input_grad_kernel<double>(
        double *__restrict__ inputs_grad,
        const double *__restrict__ aligned_inputs_grad,
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
);


}
