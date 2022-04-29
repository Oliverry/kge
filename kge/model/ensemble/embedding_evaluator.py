import copy
from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn

from kge import Configurable, Config, Dataset
from kge.misc import init_from


class EmbeddingEvaluator(nn.Module, Configurable):
    """
    Base class for the combination of metaembeddings to an overall score.
    """

    def __init__(self, config: Config, configuration_key, parent_configuration_key):
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)

        # Fetch entity and relation dimensionality for evaluator
        self.entity_dim = config.get(parent_configuration_key + ".entity_agg_dim")
        self.relation_dim = config.get(parent_configuration_key + ".relation_agg_dim")

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        """
        Computes a score for the given subject, predicate and object embeddings.
        The tensor have the form n_s times E, n_p times E and n_o times E, where n_i is the respective number of tensor
        for each embedding target and E is the metaembedding size.
        Then the embeddings are combined specified by the combine statement.
        :param s: Tensor of subject embeddings.
        :param p: Tensor of predicate embeddings.
        :param o: Tensor of object embeddings.
        :param combine: String to specify how embeddings shall be combined.
        :return: Tensor of scores.
        """
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        return []


class KgeAdapter(EmbeddingEvaluator):
    """
    An adapter class to use the scoring function of implemented KGE models.
    """

    def __init__(self, dataset: Dataset, config: Config, parent_configuration_key):
        EmbeddingEvaluator.__init__(
            self, config, "kge_adapter", parent_configuration_key
        )

        model_name = self.get_option("model.type")
        class_name = config.get(model_name + ".class_name")

        # change embedding size to aggregated dimensions
        dim_options = {
            self.configuration_key + ".model.entity_embedder.dim": self.entity_dim,
            self.configuration_key + ".model.relation_embedder.dim": self.relation_dim,
        }
        self.config.load_options(dim_options, create=True)

        # try to create model
        try:
            self.model = init_from(
                class_name,
                config.get("modules"),
                config=config,
                dataset=dataset,
                configuration_key=self.configuration_key + ".model",
                init_for_load_only=True,
            )
            self.model.to(config.get("job.device"))
        except:
            config.log(f"Failed to create model {model_name} (class {class_name}).")
            raise

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        res = self.model.get_scorer().score_emb(s, p, o, combine=combine)
        if combine == "spo":
            res = res.view(-1)
        return res


class FineTuning(EmbeddingEvaluator):
    """
    A finetuning model that uses separate neural networks for entities and relations, which transform metaembeddings
    before applying a KGE adapter to compute a final score.
    """

    def __init__(self, dataset: Dataset, config: Config, parent_configuration_key):
        EmbeddingEvaluator.__init__(
            self, config, "finetuning", parent_configuration_key
        )

        num_layers = self.get_option("num_layers")
        dropout = self.get_option("dropout")

        # create the entity finetuning layers
        i = 0
        entity_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            entity_nn_dict[str(i) + "-dropout"] = nn.Dropout(p=dropout)
            i += 1
            entity_nn_dict[str(i) + "-linear"] = nn.Linear(
                self.entity_dim, self.entity_dim
            )
            i += 1
            if idx + 1 < num_layers:
                entity_nn_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1
        self.entity_finetuner = torch.nn.Sequential(entity_nn_dict)

        # create the relation finetuning layers
        relation_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            relation_nn_dict[str(i) + "-dropout"] = nn.Dropout(p=dropout)
            i += 1
            relation_nn_dict[str(i) + "-linear"] = nn.Linear(
                self.relation_dim, self.relation_dim
            )
            i += 1
            if idx + 1 < num_layers:
                relation_nn_dict[str(i) + "-relu"] = nn.ReLU()
                i += 1
        self.relation_finetuner = torch.nn.Sequential(relation_nn_dict)

        # create the adapter
        self.adapter = KgeAdapter(dataset, config, parent_configuration_key)

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        s_finetuned = self.entity_finetuner(s)
        p_finetuned = self.relation_finetuner(p)
        o_finetuned = self.entity_finetuner(o)
        res = self.adapter.score_emb(s_finetuned, p_finetuned, o_finetuned, combine)
        return res
