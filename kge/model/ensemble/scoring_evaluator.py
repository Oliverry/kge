import torch
from torch import Tensor, nn

from kge import Configurable, Config


class ScoringEvaluator(torch.nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def forward(self, scores: Tensor) -> Tensor:
        """
        Takes a tensor of scores of the form n times E, where n is the number of spo scores
        and E is the number of ensemble approaches.
        Then the scores are combined row wise
        :param scores: Tensor of scores
        :return: Combined tensor of scores
        """
        raise NotImplementedError


class AvgScoringEvaluator(ScoringEvaluator):

    def __init__(self, config: Config, configuration_key=None):
        super().__init__(config, configuration_key)

    def forward(self, scores: Tensor) -> Tensor:
        res = torch.mean(scores, dim=1)
        return res


class PlattScalingEvaluator(ScoringEvaluator):

    def __init__(self, config: Config, configuration_key=None):
        super().__init__(config, configuration_key)
        m = 2  # int(self.get_option("num_models"))
        self.scalers = nn.ModuleList([PlattScaler(config, configuration_key) for _ in range(0, m)])

    def forward(self, scores: Tensor) -> Tensor:
        states = [scaler.state_dict() for scaler in self.scalers]
        res = None
        t = torch.transpose(scores, 0, 1)
        for idx, scaler in enumerate(self.scalers):
            tmp = t[idx]
            tmp = torch.unsqueeze(tmp, dim=-1)
            tmp = self.scalers[idx](tmp)
            if res is None:
                res = tmp
            else:
                res = torch.cat((res, tmp), 1)
        res = torch.mean(res, dim=1)
        return res


class PlattScaler(nn.Module):

    def __init__(self, config: Config, configuration_key):
        super(PlattScaler, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores: Tensor) -> Tensor:
        t = self.linear(scores)
        t = self.sigmoid(-t)
        return t

