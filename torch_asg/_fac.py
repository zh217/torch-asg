import torch.autograd


class FAC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target):
        # C: # chars
        # T: input length
        # N: batch size
        # S: target length
        # targets    N * S
        # inputs     T * N * C
        # result     N
        # scale      N
        # alpha      T * N * S
        # transition C * C
        # trans_next N * S
        # trans_self N * S

        # prepare trans_next and trans_self (necessary??)
        # alpha[0, n, _] <- -inf
        # alpha[0, n, 0] <- inputs[0, n, targets[n, 0]]
        # iterate over t <- T:
        #   s1 <- trans_self[n, s] + alpha[t - 1, n, s]
        #   s2 <- trans_next[n, s] + alpha[t - 1, n, s - 1]
        #   alpha[t, n, s] <- inputs[t, n, targets[n, s]] + logadd(s1, s2)

        pass

    @staticmethod
    def backward(ctx, grad_loss):
        pass
