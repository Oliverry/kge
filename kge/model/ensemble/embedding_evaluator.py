import copy

import torch
from torch import Tensor

from kge import Configurable, Config
from kge.model.kge_model import KgeModel


class EmbeddingEvaluator(torch.nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor) -> Tensor:
        """
        Takes tensors of embeddings of the form n times E, where n is the number of triples
        and E is dimensionality of the embeddings.
        Then the embeddings are combined row wise.
        :param s:
        :param p:
        :param o:
        :return:
        """
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self, savepoint):
        raise NotImplementedError


class KgeAdapter(EmbeddingEvaluator):

    def __init__(self, dataset, config: Config, configuration_key=None):
        EmbeddingEvaluator.__init__(self, config, configuration_key)
        if config.get("job.type") == "train":
            model_name = self.get_option("model")
            model_options = {"model": model_name,
                             model_name: copy.deepcopy(config.options["kge_adapter"][model_name]),
                             "job.device": config.get("job.device")}
            model_config = Config()
            model_config.load_options(model_options, create=True)
            self.model = KgeModel.create(model_config, dataset)
        else:
            pass  # self.model = KgeModel.create_from(None, dataset), better postpone to load

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        return self.model.get_scorer().score_emb(s, p, o, combine="spo").view(-1)

    def save(self):
        pass

    def load(self, savepoint):
        pass
