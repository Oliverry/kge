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
        self.linear1 = nn.Linear(dim_in, layer_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(layer_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def score_emb_spo(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor) -> Tensor:
        embeds = torch.cat((s_emb, p_emb, o_emb), 1)
        embeds = self.linear1(embeds)
        embeds = self.tanh(embeds)
        embeds = self.linear2(embeds)
        scores = self.sigmoid(embeds)
        return scores

    def score_emb(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str) -> Tensor:
        n = p_emb.size(0)
        if combine == "spo":
            embeds = torch.cat((s_emb, p_emb, o_emb), 1)
            embeds = self.linear1(embeds)
            embeds = self.tanh(embeds)
            embeds = self.linear2(embeds)
            out = self.sigmoid(embeds)
        elif combine == "sp_":
            score_list = []
            for o_emb_single in o_emb:
                o_emb_rep = o_emb_single.repeat((n, 1))
                embeds = torch.cat((s_emb, p_emb, o_emb_rep), 1)
                embeds = self.linear1(embeds)
                embeds = self.tanh(embeds)
                embeds = self.linear2(embeds)
                embeds_score = self.sigmoid(embeds)
                score_list.append(embeds_score)
            out = torch.cat(score_list, 1)
        elif combine == "_po":
            score_list = []
            for s_emb_single in s_emb:
                s_emb_rep = s_emb_single.repeat((n, 1))
                embeds = torch.cat((s_emb_rep, p_emb, o_emb), 1)
                embeds = self.linear1(embeds)
                embeds = self.tanh(embeds)
                embeds = self.linear2(embeds)
                embeds_score = self.sigmoid(embeds)
                score_list.append(embeds_score)
            out = torch.cat(score_list, 1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


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
