import torch
from torch import Tensor, nn

from kge import Configurable, Config


class ScoringEvaluator(torch.nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def forward(self, scores: Tensor) -> Tensor:
        raise NotImplementedError


class AvgScoringEvaluator(ScoringEvaluator):

    def __init__(self, config: Config, configuration_key=None):
        super().__init__(config, configuration_key)

    def forward(self, scores: Tensor) -> Tensor:
        return torch.mean(scores, dim=0)


class PlattScalingEvaluator(ScoringEvaluator):

    def __init__(self, config: Config, configuration_key=None):
        super().__init__(config, configuration_key)
        m = 2  # int(self.get_option("num_models"))
        self.scalers = [PlattScaler(config, configuration_key) for _ in range(0, m)]

    def forward(self, scores: Tensor) -> Tensor:
        res = None
        for idx, scaler in enumerate(self.scalers):
            tmp = scores[idx]
            tmp = torch.unsqueeze(tmp, dim=-1)
            tmp = self.scalers[idx].forward(tmp)
            if res is None:
                res = tmp
            else:
                res = torch.cat((res, tmp), 1)
        res = torch.mean(res, dim=1)
        return res


class PlattScaler:

    def __init__(self, config: Config, configuration_key) -> Tensor:
        self.linear = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores: Tensor) -> Tensor:
        t = self.linear(scores)
        t = self.sigmoid(-t)
        return t

