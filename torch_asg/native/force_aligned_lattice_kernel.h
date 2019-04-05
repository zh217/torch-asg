//
// Created by amade on 4/5/2019.
//

#ifndef TORCH_ASG_FORCE_ALIGNED_LATTICE_KERNEL_H
#define TORCH_ASG_FORCE_ALIGNED_LATTICE_KERNEL_H


namespace torch_asg {


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
);

template<typename scalar_t>
at::Tensor
make_aligned_transition_gpu(
        at::Tensor &transition,
        at::Tensor &outputs,
        at::Tensor &input_lengths,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t batch_output_len
);

template<typename scalar_t>
at::Tensor
collect_transition_grad_gpu(
        at::Tensor &aligned_transition_grad,
        at::Tensor &outputs,
        at::Tensor &output_lengths,
        int64_t num_batches,
        int64_t num_labels
);

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
);

}

#endif //TORCH_ASG_FORCE_ALIGNED_LATTICE_KERNEL_H
