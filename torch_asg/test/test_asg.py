import pytest
import torch
from torch.autograd import gradcheck
from torch_asg import ASGLoss
from torch_asg.asg import FCC, FAC

TEST_CUDA_OPTS = [False, True] if torch.cuda.is_available() else []


# @pytest.mark.skip()
@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_run(cuda):
    num_labels = 7
    input_batch_len = 6
    num_batches = 2
    target_batch_len = 5
    asg_loss = ASGLoss(num_labels=num_labels,
                       reduction='mean',  # mean (default), sum, none
                       )

    inputs = torch.randn(input_batch_len, num_batches, num_labels, requires_grad=True)
    targets = torch.randint(0, num_labels, (num_batches, target_batch_len))
    input_lengths = torch.randint(1, input_batch_len + 1, (num_batches,))
    target_lengths = torch.randint(1, target_batch_len + 1, (num_batches,))

    if cuda:
        inputs = inputs.clone().detach().cuda()
        inputs.requires_grad = True
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()
        asg_loss = asg_loss.cuda()

    asg_loss.transition.data.uniform_()

    # print('start')

    loss = asg_loss.forward(inputs, targets, input_lengths, target_lengths)
    # print('loss', loss)
    # You can get the transition matrix if you need it.
    # transition[i, j] is transition score from label j to label i.
    # print('transition matrix', asg_loss.transition)
    loss.backward()
    # print('transition matrix grad', asg_loss.transition.grad)
    # print('inputs grad', inputs.grad)



@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fcc_1(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 1
    N = 2
    inputs = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(B, T, N).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    inputs = torch.log(inputs)
    targets = torch.zeros(2, 1, dtype=torch.long)

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()

    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S]))
    assert results.abs().sum() < EPSILON
    gradcheck(
        lambda inp, trans: FCC.apply(trans, inp, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S])).sum(),
        (inputs.clone().detach().requires_grad_(True),
         transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fcc_2(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 1
    N = 4
    inputs = torch.full((B, T, N), torch.log(torch.tensor(0.25))).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([1, 2]).view(2, 1)

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()

    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S]))
    assert results.abs().sum() < EPSILON, results.abs().sum()
    gradcheck(
        lambda inp, trans: FCC.apply(trans, inp, targets, torch.LongTensor([T, T]), torch.LongTensor([S, S])).sum(),
        (inputs.clone().detach().requires_grad_(True),
         transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
@pytest.mark.parametrize('use_double', [False, True])
def test_fcc_3(cuda, use_double):
    if use_double:
        torch.set_default_dtype(torch.float64)

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

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()

    results = FCC.apply(transition, inputs, targets, torch.LongTensor([T] * B), torch.LongTensor([S] * B))
    assert results.abs().sum() < EPSILON, results.abs().sum()
    # gradcheck(
    #     lambda inp, trans: FCC.apply(trans, inp, targets, torch.LongTensor([T] * B), torch.LongTensor([S] * B)),
    #     (inputs.clone().detach().requires_grad_(True),
    #      transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fcc_grad(cuda):
    torch.set_default_dtype(torch.float64)
    B = 2
    T = 8
    S = 1
    N = 3
    inputs = torch.empty((T, B, N)).uniform_()
    targets = torch.randint(0, N, (B, S))
    transition = torch.empty((N, N)).uniform_()

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()

    def f(inputs, transition):
        return FCC.apply(transition, inputs, targets, torch.LongTensor([T] * B), torch.LongTensor([S] * B)).sum()

    gradcheck(f,
              (inputs.clone().detach().requires_grad_(True),
               transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fcc_grad2(cuda):
    torch.set_default_dtype(torch.float64)
    num_labels = 7
    input_batch_len = 6
    num_batches = 2
    target_batch_len = 5
    for i in range(1):
        inputs = torch.randn(input_batch_len, num_batches, num_labels, requires_grad=True)
        targets = torch.randint(0, num_labels, (num_batches, target_batch_len))
        transition = torch.empty((num_labels, num_labels)).uniform_()
        input_lengths = torch.randint(1, input_batch_len + 1, (num_batches,))
        target_lengths = torch.randint(1, target_batch_len + 1, (num_batches,))

        # r = FCC.apply(transition, inputs, targets,
        #               input_lengths, target_lengths,
        #               'target_size_sqrt').sum()
        # print(r)
        # r.backward()

        if cuda:
            inputs = inputs.cuda()
            transition = transition.cuda()
            targets = targets.cuda()

        def f(inputs, transition):
            return FCC.apply(transition, inputs, targets,
                             input_lengths, target_lengths).sum()

        gradcheck(f,
                  (inputs.clone().detach().requires_grad_(True),
                   transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fac_1(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 2
    N = 2
    inputs = torch.tensor([1.0, 0.0,
                           0.0, 1.0,
                           0.5, 0.5,

                           1.0, 0.0,
                           0.0, 1.0,
                           0.0, 1.0]).view(B, T, N).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([0, 1,
                                0, 1]).view(B, S)
    expected = torch.logsumexp(torch.tensor([[1.5, 2.5], [2., 3.]]), dim=-1)
    input_lengths = torch.LongTensor([T, T])
    target_lengths = torch.LongTensor([S, S])

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()
        expected = expected.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    results = FAC.apply(transition, inputs, targets, input_lengths, target_lengths)
    assert (results - expected).abs().sum() < EPSILON, results.abs().sum()
    gradcheck(
        lambda inp, trans: FAC.apply(trans, inp, targets, input_lengths, target_lengths).sum(),
        (inputs.clone().detach().requires_grad_(True),
         transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fac_2(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 1
    T = 3
    S = 2
    N = 4
    inputs = torch.full((B, T, N), torch.log(torch.tensor(0.25))).permute(1, 0, 2)
    transition = torch.zeros(N, N)
    targets = torch.LongTensor([0, 1]).view(B, S)
    input_lengths = torch.LongTensor([T])
    target_lengths = torch.LongTensor([S])

    if cuda:
        inputs = inputs.cuda()
        transition = transition.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    results = FAC.apply(transition, inputs, targets, input_lengths, target_lengths)
    expected = -torch.log(torch.tensor(32.))
    assert (results - expected).abs().sum() < EPSILON, results.abs().sum()
    gradcheck(
        lambda inp, trans: FAC.apply(trans, inp, targets, input_lengths, target_lengths).sum(),
        (inputs.clone().detach().requires_grad_(True),
         transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_fac_grad(cuda):
    torch.set_default_dtype(torch.float64)
    B = 3
    T = 5
    S = 3
    N = 3
    inputs = torch.empty((T, B, N)).uniform_()
    targets = torch.tensor([[1, 2, 1],
                            [0, 1, 0],
                            [1, 0, 0], ])
    transition = torch.empty((N, N)).uniform_()
    input_lengths = torch.LongTensor([T] * B)
    target_lengths = torch.LongTensor([3, 2, 1])

    if cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        transition = transition.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    # FAC.apply(transition, inputs, targets[:B], torch.LongTensor([T] * B)[:B],
    #           torch.LongTensor([3, 2, 1][:B]),
    #           'target_size_sqrt').sum()

    def f(inputs, transition):
        return FAC.apply(transition, inputs, targets[:B], input_lengths[:B], target_lengths[:B]).sum()

    gradcheck(f,
              (inputs[:, :B].clone().detach().requires_grad_(True),
               transition.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_asg_1(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 2
    T = 3
    S = 2
    N = 2
    inputs = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(B, T, N).permute(1, 0, 2)
    inputs = torch.log(inputs)
    targets = torch.tensor([[0, 1], [0, 1]])
    input_lengths = torch.tensor([T] * B)
    output_lengths = torch.tensor([S] * B)
    expected = torch.tensor([-torch.log(torch.tensor(0.5)), 0])
    asg = ASGLoss(N, reduction='none')

    if cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        output_lengths = output_lengths.cuda()
        expected = expected.cuda()
        asg = asg.cuda()

    loss = asg.forward(inputs, targets, input_lengths, output_lengths)

    assert (loss - expected).abs().sum() < EPSILON

    gradcheck(
        lambda inp: asg.forward(inp, targets, input_lengths, output_lengths),
        (inputs.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_asg_2(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 1
    T = 3
    S = 2
    N = 4
    inputs = torch.full((T, B, N), torch.log(torch.tensor(0.25)))
    targets = torch.tensor([[0, 1]])
    asg = ASGLoss(N, reduction='mean')
    expected = torch.tensor(32.).log_()
    input_lengths, output_lengths = torch.tensor([T]), torch.tensor([S])

    if cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        output_lengths = output_lengths.cuda()
        expected = expected.cuda()
        asg = asg.cuda()

    loss = asg.forward(inputs, targets, input_lengths, output_lengths)

    assert (loss - expected).abs().sum() < EPSILON
    gradcheck(
        lambda inp: asg.forward(inp, targets, input_lengths, output_lengths),
        (inputs.clone().detach().requires_grad_(True)))


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_asg_3(cuda):
    EPSILON = 1e-10
    B = 1
    T = 3
    S1 = 4
    S2 = 3
    N = 4
    inputs = torch.full((T, B, N), torch.tensor(0.25).log_())
    targets = torch.tensor([[0, 1, 1, 1]])
    input_lengths, output_lengths = torch.tensor([T]), torch.tensor([S1])
    asg = ASGLoss(N, reduction='mean')

    if cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        output_lengths = output_lengths.cuda()
        asg = asg.cuda()

    loss1 = asg.forward(inputs, targets, input_lengths, output_lengths)
    loss2 = asg.forward(inputs, targets, input_lengths, output_lengths)
    assert (loss1 - loss2).abs().sum() < EPSILON


@pytest.mark.parametrize('cuda', TEST_CUDA_OPTS)
def test_asg_4(cuda):
    torch.set_default_dtype(torch.float64)
    EPSILON = 1e-10
    B = 3
    T = 5
    S = 5
    N = 6
    inputs_o = torch.tensor([
        -0.4340, -0.0254, +0.3667, +0.4180, -0.3805, -0.1707,
        +0.1060, +0.3631, -0.1122, -0.3825, -0.0031, -0.3801,
        +0.0443, -0.3795, +0.3194, -0.3130, +0.0094, +0.1560,
        +0.1252, +0.2877, +0.1997, -0.4554, +0.2774, -0.2526,
        -0.4001, -0.2402, +0.1295, +0.0172, +0.1805, -0.3299,

        +0.3298, -0.2259, -0.0959, +0.4909, +0.2996, -0.2543,
        -0.2863, +0.3239, -0.3988, +0.0732, -0.2107, -0.4739,
        -0.0906, +0.0480, -0.1301, +0.3975, -0.3317, -0.1967,
        +0.4372, -0.2006, +0.0094, +0.3281, +0.1873, -0.2945,
        +0.2399, +0.0320, -0.3768, -0.2849, -0.2248, +0.3186,

        +0.0225, -0.3867, -0.1929, -0.2904, -0.4958, -0.2533,
        +0.4001, -0.1517, -0.2799, -0.2915, +0.4198, +0.4506,
        +0.1446, -0.4753, -0.0711, +0.2876, -0.1851, -0.1066,
        +0.2081, -0.1190, -0.3902, -0.1668, +0.1911, -0.2848,
        -0.3846, +0.1175, +0.1052, +0.2172, -0.0362, +0.3055,
    ], requires_grad=True)
    targets = torch.tensor([
        2, 1, 5, 1, 3,
        4, 3, 5, 0, 0,
        3, 2, 2, 1, 0,
    ]).view(B, S)
    expected_loss = torch.tensor([7.7417464256287,
                                  6.4200420379639,
                                  8.2780694961548, ])
    input_lengths, output_lengths = torch.tensor([T] * B), torch.tensor([5, 3, 4])

    expected_input_grad = torch.tensor([0.1060, 0.1595, -0.7639, 0.2485, 0.1118, 0.1380,
                                        0.1915, -0.7524, 0.1539, 0.1175, 0.1717, 0.1178,
                                        0.1738, 0.1137, 0.2288, 0.1216, 0.1678, -0.8057,
                                        0.1766, -0.7923, 0.1902, 0.0988, 0.2056, 0.1210,
                                        0.1212, 0.1422, 0.2059, -0.8160, 0.2166, 0.1300,

                                        0.2029, 0.1164, 0.1325, 0.2383, -0.8032, 0.1131,
                                        0.1414, 0.2602, 0.1263, -0.3441, -0.3009, 0.1172,
                                        0.1557, 0.1788, 0.1496, -0.5498, 0.0140, 0.0516,
                                        0.2306, 0.1219, 0.1503, -0.4244, 0.1796, -0.2579,
                                        0.2149, 0.1745, 0.1160, 0.1271, 0.1350, -0.7675,

                                        0.2195, 0.1458, 0.1770, -0.8395, 0.1307, 0.1666,
                                        0.2148, 0.1237, -0.6613, -0.1223, 0.2191, 0.2259,
                                        0.2002, 0.1077, -0.8386, 0.2310, 0.1440, 0.1557,
                                        0.2197, -0.1466, -0.5742, 0.1510, 0.2160, 0.1342,
                                        0.1050, -0.8265, 0.1714, 0.1917, 0.1488, 0.2094, ]).view(B, T, N).permute(1, 0,
                                                                                                                  2)
    expected_trans_grad = torch.tensor([0.3990, 0.3396, 0.3486, 0.3922, 0.3504, 0.3155,
                                        0.3666, 0.0116, -1.6678, 0.3737, 0.3361, -0.7152,
                                        0.3468, 0.3163, -1.1583, -0.6803, 0.3216, 0.2722,
                                        0.3694, -0.6688, 0.3047, -0.8531, -0.6571, 0.2870,
                                        0.3866, 0.3321, 0.3447, 0.3664, -0.2163, 0.3039,
                                        0.3640, -0.6943, 0.2988, -0.6722, 0.3215, -0.1860, ]).view(N, N)

    asg = ASGLoss(N, reduction='none')

    if cuda:
        inputs_o = inputs_o.clone().detach().cuda()
        inputs_o.requires_grad = True
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        output_lengths = output_lengths.cuda()
        expected_loss = expected_loss.cuda()
        expected_input_grad = expected_input_grad.cuda()
        expected_trans_grad = expected_trans_grad.cuda()
        asg = asg.cuda()

    inputs = inputs_o.view(B, T, N).permute(1, 0, 2)

    loss = asg.forward(inputs, targets, input_lengths, output_lengths)
    loss.sum().backward()

    assert asg.transition.grad is not None
    assert inputs_o.grad is not None

    assert (loss - expected_loss).abs().sum() < 1e-3
    assert (expected_input_grad - inputs_o.grad.view(B, T, N).permute(1, 0, 2)).abs().max() < 1e-4
    assert (expected_trans_grad - asg.transition.grad).abs().max() < 1e-4
