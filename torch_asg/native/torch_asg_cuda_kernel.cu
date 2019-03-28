#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

#include <cub/cub.cuh>


#include "torch_asg_cuda_kernel.h"

#define GPU_ERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace torch_asg {

constexpr int kBlockSize = 32;

template<typename scalar_t>
__global__ void forward_step_template(
        int64_t input_batch_length,
        int64_t batch_size,
        int64_t num_labels,
        const scalar_t *transition_p,
        const scalar_t *inputs_p,
        const scalar_t *alpha_p,
        const scalar_t *mult_temp_p
) {
    const scalar_t neg_inf = std::numeric_limits<scalar_t>::infinity();
    using BlockReduce = cub::BlockReduce<scalar_t, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ double max_value;

    __syncthreads();
}

void forward_template(
        const at::Tensor &transition,
        const at::Tensor &inputs,
        at::Tensor &alpha,
        at::Tensor &mult_temp
) {
    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "forward_template", [&] {
        for (int64_t t = 0; t < 10; ++t) {
            forward_step_template<scalar_t><<<1,1>>>(0,
                                            0,
                                            0,
                                            transition.data<scalar_t>(),
                                            inputs.data<scalar_t>(),
                                            alpha.data<scalar_t>(),
                                            mult_temp.data<scalar_t>());
            GPU_ERRCHK(cudaGetLastError());
        }
    });
}
}
