import torch
from torch_asg import ASG
from torch_asg.asg import FCC, FAC


# def test_run():
#     C = 7
#     T = 6
#     N = 1
#     S = 5
#     asg_loss = ASG(num_labels=C)
#     for i in range(10):
#         inputs = torch.randn(T, N, C)
#         targets = torch.randint(0, C, (N, S))
#         input_lengths = torch.randint(1, T + 1, (N,))
#         target_lengths = torch.randint(1, S + 1, (N,))
#         loss = asg_loss.forward(inputs, targets, input_lengths, target_lengths)
#         print(loss)
#         loss.backward()
#         print(asg_loss.transition.grad)


def test_fcc_1():
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 1
    N = 2
    inputs = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(B, T, N).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    inputs = torch.log(inputs)
    targets = torch.zeros(2, 1, dtype=torch.long)
    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S]), 'none')
    assert results.abs().sum() < EPSILON


def test_fcc_2():
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 1
    N = 4
    inputs = torch.full((B, T, N), torch.log(torch.tensor(0.25))).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([1, 2]).view(2, 1)
    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S]), 'none')
    assert results.abs().sum() < EPSILON, results.abs().sum()


def test_fcc_3():
    EPSILON = 1e-4
    B = 3
    T = 300
    S = 50
    N = 40
    inputs = torch.empty((B, T, N)).uniform_()
    inputs = inputs / inputs.sum(dim=-1, keepdim=True)
    inputs = torch.log(inputs)
    inputs = inputs.permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.zeros((B, S), dtype=torch.long)
    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T, ] * B), torch.LongTensor([S, ] * B), 'none')
    assert results.abs().sum() < EPSILON, results.abs().sum()


def test_fac_1():
    pass

# if __name__ == '__main__':
#     test_run()
