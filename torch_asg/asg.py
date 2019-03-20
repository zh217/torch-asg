import torch
import torch.nn as nn
from ._fac import FAC
from ._fcc import FCC


class ASG(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.n_vocab = n_vocab
        # self.scale_mode = scale_mode  # none, input_size, input_size_sqrt, target_size, target_size_sqrt
        # self.trans_diag = 0.0
        self.transition = nn.Parameter(torch.ones(n_vocab, n_vocab))

    def forward(self, log_probs, targets, input_lengths, target_lengths, reduction='none', scale_mode='none'):
        return FCC.apply(log_probs, targets, self.transition) - FAC.apply(log_probs, targets, self.transition)
