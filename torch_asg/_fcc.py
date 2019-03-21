import torch.autograd


class FCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, transition):
        pass

    @staticmethod
    def backward(ctx, grad_loss):
        pass
