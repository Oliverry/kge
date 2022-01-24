import torch.nn
from torch import Tensor

from kge import Config, Dataset
from kge.model import KgeModel
from kge.model.kge_model import RelationalScorer


class SmeScorer(RelationalScorer):

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        ent_emb_dim = 256  # self.get_option("entity_embedder.dim")
        rel_emb_dim = 256  # self.get_option("relation_embedder.dim")
        self.emb_dim = self.get_option("emb_dim")
        self.sp_linear = torch.nn.Linear(ent_emb_dim + rel_emb_dim, self.emb_dim)
        self.op_linear = torch.nn.Linear(rel_emb_dim + ent_emb_dim, self.emb_dim)

    def score_emb_spo(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor) -> Tensor:
        n = s_emb.size()[0]
        sp_embed = torch.cat((s_emb, p_emb), 1)
        op_embed = torch.cat((o_emb, p_emb), 1)
        sp_embed = self.sp_linear(sp_embed)
        op_embed = self.op_linear(op_embed)
        res = torch.einsum('ij,ij->i', sp_embed, op_embed)  # compute row-wise dot product
        return res.view(n)


class SemanticMatchingEnergy(KgeModel):
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
            scorer=SmeScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
