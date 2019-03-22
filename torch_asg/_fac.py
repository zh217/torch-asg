import torch.autograd
import torch_asg_native


class FAC(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                transition,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                reduction='none',
                scale_mode='none'):
        outputs = torch_asg_native.fac_forward(log_probs,
                                               targets,
                                               input_lengths,
                                               target_lengths,
                                               transition,
                                               reduction,
                                               scale_mode)
        alpha, trans_next, trans_self, loss = outputs
        ctx.save_for_backward(alpha, trans_next, trans_self, loss)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        alpha, trans_next, trans_self, result = ctx.saved_variables
        grad_transition, grad_log_probs = torch_asg_native.fac_backward()
        return grad_transition, grad_log_probs, None, None, None, None, None, None
