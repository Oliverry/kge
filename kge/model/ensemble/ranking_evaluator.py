import torch
from torch import Tensor, nn

from kge import Configurable, Config


class RankingEvaluator(nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def forward(self, scores: Tensor) -> Tensor:
        """
        :param scores: Tensor of scores
        :return: Combined tensor of scores
        """
        raise NotImplementedError


class AvgRankingEvaluator(RankingEvaluator):

    def __init__(self, config: Config):
        super().__init__(config, None)

    def forward(self, scores: Tensor) -> Tensor:
        scores_sz = scores.size()
        res = torch.mean(scores, dim=len(scores_sz) - 1)
        return res
