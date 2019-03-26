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
        results = torch_asg_native.fcc_loss_cpu(transition, inputs, targets, input_lengths, target_lengths, scale_mode)
        out, alpha, alpha_max_contrib, scale = results
        ctx.save_for_backward(transition, inputs, targets, input_lengths,
                              target_lengths, alpha, alpha_max_contrib, scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        transition, inputs, targets, input_lengths, target_lengths, alpha, alpha_max_contrib, scale = ctx.saved_tensors
        results = torch_asg_native.fcc_loss_backward_cpu(grad_out, transition, inputs, targets, input_lengths,
                                                         target_lengths, alpha, alpha_max_contrib, scale)
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
        if self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'mean':
            return result.mean()
        else:
            return result
