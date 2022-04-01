import torch
from torch import nn, Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel


class MultiLayerPerceptronScorer(RelationalScorer):

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        entity_dim = self.get_option("entity_embedder.dim")
        relation_dim = self.get_option("relation_embedder.dim")
        dim_in = 2 * entity_dim + relation_dim
        layer_dim = self.get_option("layer_dim")
        if layer_dim < 0:
            layer_dim = entity_dim
        self.linear1 = nn.Linear(dim_in, layer_dim)  # embedding concatenation (spo) -> entitiy embedding size
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(layer_dim, 1)  # entity embedding size -> 1
        self.sigmoid = nn.Sigmoid()

    def score_emb_spo(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor) -> Tensor:
        embeds = torch.cat((s_emb, p_emb, o_emb), 1)
        embeds = self.linear1(embeds)
        embeds = self.tanh(embeds)
        embeds = self.linear2(embeds)
        scores = self.sigmoid(embeds)
        return scores


class MultilayerPerceptron(KgeModel):

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key=None,
            init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=MultiLayerPerceptronScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)
