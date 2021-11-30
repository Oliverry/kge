import torch
from torch import Tensor

from kge import Configurable, Config


class EmbeddingEvaluator(torch.nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def forward(self, embeds: Tensor) -> Tensor:
        """
        Takes a tensor of embeddings of the form n times E, where n is the number of triples
        and E is the number of ensemble approaches.
        Then the scores are combined row wise
        :param scores: Tensor of scores
        :return: Combined tensor of scores
        """
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self, savepoint):
        raise NotImplementedError