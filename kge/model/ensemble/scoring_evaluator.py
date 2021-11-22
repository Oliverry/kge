import torch
from torch import nn, Tensor


class ScoringEvaluator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores: Tensor) -> Tensor:
        raise NotImplementedError


class AvgScoringEvaluator(ScoringEvaluator):

    def __init__(self):
        super().__init__()

    def forward(self, scores: Tensor) -> Tensor:
        return torch.mean(scores, dim=0)
