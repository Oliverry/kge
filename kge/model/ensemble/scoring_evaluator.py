from typing import List

import torch
from torch import Tensor, nn

from kge import Configurable, Config


class ScoringEvaluator(nn.Module, Configurable):
    """
    Base class for all scoring evaluators.
    """

    def __init__(self, config: Config, configuration_key):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def forward(self, scores: Tensor) -> Tensor:
        """
        Combines the scores by the function of the specific evaluator.
        :param scores: Tensor of scores.
        :return: Combined tensor of scores.
        """
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        return []


class AvgScoringEvaluator(ScoringEvaluator):
    """
    Evaluator that combines the scores by taking the mean.
    """

    def __init__(self, config: Config):
        super().__init__(config, None)

    def forward(self, scores: Tensor) -> Tensor:
        scores_sz = scores.size()
        res = torch.mean(scores, dim=len(scores_sz) - 1)
        return res


class PlattScalingEvaluator(ScoringEvaluator):
    """
    Evaluator that uses a platt scaler for each KGE model for balancing before taking the mean.
    """

    def __init__(self, config: Config, parent_configuration_key):
        super().__init__(config, None)

        # create the model specific platt scalers
        num_models = len(config.get(parent_configuration_key + ".base_models"))
        self.scalers = nn.ModuleList([PlattScaler() for _ in range(0, num_models)])

    def forward(self, scores: Tensor) -> Tensor:
        scores_len = len(scores.size())
        scores_list = []
        t = torch.transpose(scores, scores_len - 2, scores_len - 1)
        for idx, scaler in enumerate(self.scalers):
            scaler_scores = t[..., idx, :]
            scaler_scores = torch.unsqueeze(scaler_scores, dim=-1)
            scaler_scores = self.scalers[idx](scaler_scores)
            scores_list.append(scaler_scores)
        scores_comb = torch.cat(scores_list, dim=scores_len - 1)
        res = torch.mean(scores_comb, dim=scores_len - 1)
        return res


class PlattScaler(nn.Module):
    """
    A single platt scaler to transform a score into a probability after balancing.
    """

    def __init__(self):
        super(PlattScaler, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores: Tensor) -> Tensor:
        t = self.linear(scores)
        t = self.sigmoid(t)
        return t
