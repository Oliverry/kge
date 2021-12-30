import torch
from torch import nn, Tensor

from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel


class MultiLayerPerceptronScorer(RelationalScorer):

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        dim_in = self.get_option("dim_in")
        dim_t = self.get_option("dim_t")
        self._norm = self.get_option("l_norm")
        self.linear1 = nn.Linear(dim_in, dim_t)  # embedding concatenation (spo) -> entitiy embedding size
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(dim_t, 1)  # entity embedding size -> 1
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
