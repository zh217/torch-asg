import torch
import torch.autograd
import torch.nn as nn
import torch_asg_native


class FAC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, inputs, targets, input_lengths, target_lengths):
        batch_input_len, num_batches, num_labels = inputs.shape
        _, batch_output_len = targets.shape
        results = torch_asg_native.force_aligned_forward(inputs,
                                                         targets,
                                                         transition,
                                                         input_lengths,
                                                         target_lengths,
                                                         batch_input_len,
                                                         num_batches,
                                                         num_labels,
                                                         batch_output_len)
        scores, alpha, beta, path_contrib = results
        ctx.save_for_backward(alpha, beta, path_contrib, targets, input_lengths, target_lengths, transition)
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        alpha, beta, path_contrib, targets, input_lengths, target_lengths, transition = ctx.saved_tensors
        batch_input_len, num_batches, batch_output_len = alpha.shape
        num_labels, _ = transition.shape
        results = torch_asg_native.force_aligned_backward(grad_out, alpha, beta, path_contrib,
                                                          targets, input_lengths, target_lengths,
                                                          batch_input_len, num_batches, num_labels, batch_output_len)
        grad_transition, grad_inputs = results
        return grad_transition, grad_inputs, None, None, None, None, None


class FCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, inputs, targets, input_lengths, target_lengths):
        input_batch_len, num_batches, num_labels = inputs.shape
        scores, alpha, beta, path_contrib = torch_asg_native.fully_connected_forward(inputs, transition, input_lengths,
                                                                                     input_batch_len, num_batches,
                                                                                     num_labels)
        ctx.save_for_backward(alpha, beta, path_contrib)
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        alpha, beta, path_contrib = ctx.saved_tensors
        input_batch_len, num_batches, num_labels = alpha.shape
        results = torch_asg_native.fully_connected_backward(grad_out, alpha, beta, path_contrib, input_batch_len,
                                                            num_batches,
                                                            num_labels)
        grad_transition, grad_inputs = results
        return grad_transition, grad_inputs, None, None, None, None, None


class ASGGPUFastForwardOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, outputs, transition, input_lengths, output_lengths):
        batch_input_len, num_batches, num_labels = inputs.shape
        _, batch_output_len = outputs.shape
        return torch_asg_native.fast_asg_gpu_forward_only(inputs, outputs, transition, input_lengths, output_lengths,
                                                          batch_input_len, num_batches, num_labels, batch_output_len)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None


class ASGGPUFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, transition, outputs, input_lengths, output_lengths):
        batch_input_len, num_batches, num_labels = inputs.shape
        _, batch_output_len = outputs.shape
        results = torch_asg_native.fast_asg_gpu_forward(inputs, outputs, transition, input_lengths, output_lengths,
                                                        batch_input_len, num_batches, num_labels, batch_output_len)
        full_scores, aligned_scores, full_gamma, aligned_gamma, full_path_contrib, aligned_path_contrib = results
        ctx.save_for_backward(full_gamma, aligned_gamma,
                              full_path_contrib, aligned_path_contrib,
                              outputs, input_lengths, output_lengths)
        return full_scores, aligned_scores

    @staticmethod
    def backward(ctx, grad_full, grad_aligned):
        full_gamma, aligned_gamma, full_path_contrib, aligned_path_contrib, outputs, input_lengths, output_lengths = ctx.saved_tensors

        batch_input_len, num_batches, num_labels = full_gamma.shape
        _, batch_output_len = outputs.shape

        results = torch_asg_native.fast_asg_gpu_backward(grad_full, grad_aligned,
                                                         full_gamma, aligned_gamma,
                                                         full_path_contrib, aligned_path_contrib,
                                                         outputs, input_lengths, output_lengths,
                                                         batch_input_len, num_batches, num_labels, batch_output_len)
        grad_transition, grad_inputs = results
        return grad_inputs, grad_transition, None, None, None


class ASGLoss(nn.Module):
    def __init__(self, num_labels, reduction='mean', forward_only=False, gpu_no_stream_impl=False):
        super().__init__()
        self.num_labels = num_labels
        self.reduction = reduction  # mean, sum, none
        self.transition = nn.Parameter(torch.zeros(num_labels, num_labels))
        self.forward_only = forward_only
        self.gpu_no_stream_impl = gpu_no_stream_impl

    def forward(self, inputs, targets, input_lengths, target_lengths):
        batch_input_len, num_batches, num_labels = inputs.shape
        _, batch_output_len = targets.shape
        if batch_output_len > batch_input_len:
            batch_output_len = batch_input_len
            targets = targets[:, :batch_output_len]
            target_lengths = torch.min(target_lengths, other=target_lengths.new_tensor([batch_output_len]))

        if self.gpu_no_stream_impl or not inputs.is_cuda:
            # use the "serial" implementation
            fac_result = FAC.apply(self.transition, inputs, targets, input_lengths, target_lengths)
            fcc_result = FCC.apply(self.transition, inputs, targets, input_lengths, target_lengths)
            result = fcc_result - fac_result
        elif self.forward_only:
            # use the GPU fast implementation without backward support
            result = ASGGPUFastForwardOnly.apply(inputs, targets, self.transition, input_lengths, target_lengths)
        else:
            # use the fast GPU implementation
            full_scores, aligned_scores = ASGGPUFast.apply(inputs, self.transition, targets, input_lengths,
                                                           target_lengths)
            result = full_scores - aligned_scores
        if self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'mean':
            return result.mean()
        else:
            return result
