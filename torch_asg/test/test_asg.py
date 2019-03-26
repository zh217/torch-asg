import torch
from torch_asg import ASGLoss
from torch_asg.asg import FCC, FAC


def test_run():
    num_labels = 7
    input_batch_len = 6
    num_batches = 2
    target_batch_len = 5
    asg_loss = ASGLoss(num_labels=num_labels,
                       reduction='mean',  # mean (default), sum, none
                       scale_mode='none'  # none (default), input_size, input_size_sqrt, target_size, target_size_sqrt
                       )
    for i in range(1):
        inputs = torch.randn(input_batch_len, num_batches, num_labels, requires_grad=True)
        targets = torch.randint(0, num_labels, (num_batches, target_batch_len))
        input_lengths = torch.randint(1, input_batch_len + 1, (num_batches,))
        target_lengths = torch.randint(1, target_batch_len + 1, (num_batches,))
        loss = asg_loss.forward(inputs, targets, input_lengths, target_lengths)
        print('loss', loss)
        # You can get the transition matrix if you need it.
        # transition[i, j] is transition score from label j to label i.
        print('transition matrix', asg_loss.transition)
        loss.backward()
        print('transition matrix grad', asg_loss.transition.grad)
        print('inputs grad', inputs.grad)


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
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 2
    N = 2
    inputs = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(B, T, N).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([0, 1, 0, 1]).view(B, S)
    results = FAC.apply(transition, inputs, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S]), 'none')
    expected = torch.logsumexp(torch.tensor([[1.5, 2.5], [2., 3.]]), dim=-1)
    assert (results - expected).abs().sum() < EPSILON, results.abs().sum()


def test_fac_2():
    EPSILON = 1e-10
    B = 1
    T = 3
    S = 2
    N = 4
    inputs = torch.full((B, T, N), torch.log(torch.tensor(0.25))).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([0, 1]).view(B, S)
    results = FAC.apply(transition, inputs, targets, torch.LongTensor([T, ]), torch.LongTensor([S, ]), 'none')
    expected = -torch.log(torch.tensor(32.))
    assert (results - expected).abs().sum() < EPSILON, results.abs().sum()

# if __name__ == '__main__':
#     test_run()
