import torch
import torch.autograd
import torch.nn as nn
import torch_asg_native


class FAC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, inputs, targets, input_lengths, target_lengths, scale_mode):
        results = torch_asg_native.fac_loss_cpu(transition, inputs, targets, input_lengths, target_lengths, scale_mode)
        out, alpha, scale, self_trans, next_trans = results
        ctx.save_for_backward(inputs, targets, input_lengths, target_lengths, alpha, scale, self_trans, next_trans)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        inputs, targets, input_lengths, target_lengths, alpha, scale, self_trans, next_trans = ctx.saved_tensors
        results = torch_asg_native.fac_loss_backward_cpu(grad_out, inputs, targets, input_lengths, target_lengths,
                                                         alpha, scale, self_trans, next_trans)
        grad_transition, grad_inputs = results
        return grad_transition, grad_inputs, None, None, None, None, None


class FCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, inputs, targets, input_lengths, target_lengths, scale_mode):
        input_batch_len, num_batches, num_labels = inputs.shape
        scores, alpha, beta, path_contrib = torch_asg_native.fully_connected_forward(inputs, transition, input_lengths,
                                                                                     input_batch_len, num_batches,
                                                                                     num_labels)
        # forward_scores, alpha, path_contrib, should_roll = results
        ctx.save_for_backward(alpha, beta, path_contrib)
        # print('gamma', gamma)
        # print('grad_input', alpha, beta)
        # print('path_contrib', path_contrib)
        # print('path_contrib_s', path_contrib.exp())
        # print('path_contrib_s', path_contrib.softmax(3))
        # print('scores', scores)
        # print('scores from alpha', alpha[-1].logsumexp(1))
        # print('scores from beta', (beta[0] + inputs[0]).logsumexp(1))
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        alpha, beta, path_contrib = ctx.saved_tensors
        input_batch_len, num_batches, num_labels = alpha.shape
        results = torch_asg_native.fully_connected_backward(grad_out, alpha, beta, path_contrib, input_batch_len, num_batches,
                                                            num_labels)
        grad_transition, grad_inputs = results
        return grad_transition, grad_inputs, None, None, None, None, None


class ASGLoss(nn.Module):
    def __init__(self, num_labels, scale_mode='none', reduction='mean'):
        super().__init__()
        self.num_labels = num_labels
        self.scale_mode = scale_mode  # none, input_size, input_size_sqrt, target_size, target_size_sqrt
        self.reduction = reduction  # mean, sum, none
        self.transition = nn.Parameter(torch.zeros(num_labels, num_labels))

    def forward(self, inputs, targets, input_lengths, target_lengths):
        fac_result = FAC.apply(self.transition, inputs, targets, input_lengths, target_lengths, self.scale_mode)
        fcc_result = FCC.apply(self.transition, inputs, targets, input_lengths, target_lengths, self.scale_mode)
        result = fcc_result - fac_result
        # result = fac_result
        if self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'mean':
            return result.mean()
        else:
            return result
