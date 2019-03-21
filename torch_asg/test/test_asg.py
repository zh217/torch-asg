import torch
from torch_asg.asg import ASG


def test_run():
    C = 10
    T = 50
    N = 2
    S = 5
    asg_loss = ASG(n_vocab=C)
    asg_loss.forward(log_probs=torch.randn(T, N, C),
                     targets=torch.randint(0, C, (N, S)),
                     input_lengths=torch.randint(1, T + 1, (N,)),
                     target_lengths=torch.randint(1, S + 1, (N,)))
