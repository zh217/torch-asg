import torch
from torch_asg import ASG


def test_run():
    C = 7
    T = 6
    N = 1
    S = 5
    asg_loss = ASG(num_labels=C)
    for i in range(10):
        inputs = torch.randn(T, N, C)
        targets = torch.randint(0, C, (N, S))
        input_lengths = torch.randint(1, T + 1, (N,))
        target_lengths = torch.randint(1, S + 1, (N,))
        loss = asg_loss.forward(inputs, targets, input_lengths, target_lengths)
        print(loss)
        loss.backward()
        print(asg_loss.transition.grad)


if __name__ == '__main__':
    test_run()
