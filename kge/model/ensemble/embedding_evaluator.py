import copy
from collections import OrderedDict

import torch
from torch import Tensor, nn

from kge import Configurable, Config, Dataset
from kge.misc import init_from


class EmbeddingEvaluator(nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key, parent_configuration_key):
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)
        # Fetch entity and relation dimensionality for evaluator
        self.entity_dim = config.get(parent_configuration_key + ".entities.reduced_dim")
        self.relation_dim = config.get(parent_configuration_key + ".relations.reduced_dim")

    def score_emb(self, s: Tensor, p: Tensor, o: Tensor, combine: str) -> Tensor:
        """
        Takes tensors of embeddings of the form n times E, where n is the number of triples
        and E is dimensionality of the embeddings.
        Then the embeddings are combined specified by the combine statement.
        :param combine:
        :param s:
        :param p:
        :param o:
        :return:
        """
        raise NotImplementedError


class KgeAdapter(EmbeddingEvaluator):

    def __init__(self, dataset: Dataset, config: Config, parent_configuration_key):
        EmbeddingEvaluator.__init__(self, config, "kge_adapter", parent_configuration_key)

        # create model specific configuration
        model_config = Config()
        model_config.folder = config.folder
        model_config.log_folder = config.log_folder
        model_config.log_prefix = config.log_prefix
        model_name = self.get_option("model")
        model_options = {"model": model_name,
                         model_name: copy.deepcopy(config.options["kge_adapter"][model_name])}
        model_config.load_options(model_options)
        class_name = model_config.get(model_name + ".class_name")

        # if embedders are used, change embedding size to aggregated dimensions
        if "entity_embedder" in model_config.options[model_name]:
            model_config.set(model_name + ".entity_embedder.dim", self.entity_dim, create=True)
        if "relation_embedder" in model_config.options[model_name]:
            model_config.set(model_name + ".relation_embedder.dim", self.relation_dim, create=True)

        # try to create model
        try:
            self.model = init_from(
                class_name,
                config.get("modules"),
                config=model_config,
                dataset=dataset,
                configuration_key=None,
                init_for_load_only=True
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
    Finetuning model uses separate neural networks for entities and relations.
    The dimensionality of the output equals the dimensionality of the input
    KgeAdapter is used to apply the scoring function.
    """

    def __init__(self, dataset: Dataset, config: Config, parent_configuration_key):
        EmbeddingEvaluator.__init__(self, config, "finetuning", parent_configuration_key)

        num_layers = self.get_option("num_layers")

        entity_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            entity_nn_dict["linear" + str(idx)] = nn.Linear(self.entity_dim, self.entity_dim)
            if idx + 1 < num_layers:
                entity_nn_dict["relu" + str(idx)] = nn.ReLU()
        self.entity_finetuner = torch.nn.Sequential(entity_nn_dict)

        relation_nn_dict = OrderedDict()
        for idx in range(0, num_layers):
            relation_nn_dict["linear" + str(idx)] = nn.Linear(self.relation_dim, self.relation_dim)
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
