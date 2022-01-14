import copy
from collections import OrderedDict

import torch
from torch import Tensor, nn

from kge import Configurable, Config, Dataset
from kge.model.kge_model import KgeModel


class EmbeddingEvaluator(torch.nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        """
        Takes tensors of embeddings of the form n times E, where n is the number of triples
        and E is dimensionality of the embeddings.
        Then the embeddings are combined row wise.
        :param combine:
        :param s:
        :param p:
        :param o:
        :return:
        """
        raise NotImplementedError


class KgeAdapter(EmbeddingEvaluator):

    def __init__(self, dataset: Dataset, config: Config, configuration_key=None):
        EmbeddingEvaluator.__init__(self, config, configuration_key)
        model_name = self.get_option("model")
        model_options = {"model": model_name,
                         model_name: copy.deepcopy(config.options["kge_adapter"][model_name]),
                         "job.device": config.get("job.device")}
        model_config = Config()
        model_config.load_options(model_options, create=True)
        self.model = KgeModel.create(model_config, dataset)

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        res = self.model.get_scorer().score_emb(s, p, o, combine=combine)
        if combine == "spo":
            res = res.view(-1)
        return res


class FineTuning(EmbeddingEvaluator):

    def __init__(self, dataset: Dataset, config: Config, configuration_key=None):
        EmbeddingEvaluator.__init__(self, config, configuration_key)

        num_layers = self.get_option("num_layers")
        entity_dim = self.get_option("entity_dim")
        relation_dim = self.get_option("relation_dim")

        entity_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            entity_nn_dict["linear" + str(idx)] = nn.Linear(entity_dim, entity_dim)
            if idx + 1 < num_layers:
                entity_nn_dict["relu" + str(idx)] = nn.ReLU()
        self.entity_finetuner = torch.nn.Sequential(entity_nn_dict)

        relation_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            relation_nn_dict["linear" + str(idx)] = nn.Linear(relation_dim, relation_dim)
            if idx + 1 < num_layers:
                relation_nn_dict["relu" + str(idx)] = nn.ReLU()
        self.relation_finetuner = torch.nn.Sequential(relation_nn_dict)

        self.adapter = KgeAdapter(dataset, config, "kge_adapter")

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        s_finetuned = self.entity_finetuner(s)
        p_finetuned = self.relation_finetuner(p)
        o_finetuned = self.entity_finetuner(o)
        res = self.adapter.score_emb(s_finetuned, p_finetuned, o_finetuned, combine)
        return res
