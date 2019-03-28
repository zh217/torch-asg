#ifndef TORCH_ASG_TORCH_ASG_CUDA_KERNEL_H
#define TORCH_ASG_TORCH_ASG_CUDA_KERNEL_H

#include <ATen/ATen.h>

namespace torch_asg {

void forward_template(
        const at::Tensor &transition,
        const at::Tensor &inputs,
        at::Tensor &alpha,
        at::Tensor &mult_temp
);

}

#endif //TORCH_ASG_TORCH_ASG_CUDA_KERNEL_H
