import torch.nn
from torch import Tensor

from kge import Config, Dataset
from kge.model import KgeModel
from kge.model.kge_model import RelationalScorer


class SmeScorer(RelationalScorer):
    r"""Implementation of the Semantic Matching Energy KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        entity_dim = self.get_option("entity_embedder.dim")
        relation_dim = self.get_option("relation_embedder.dim")
        self.g_func = self.get_option("g_func")
        layer_dim = self.get_option("layer_dim")
        dropout = self.get_option("dropout")
        self.dropout = torch.nn.Dropout(dropout)
        if layer_dim < 0:
            layer_dim = entity_dim
        if self.g_func == "linear":
            self.sp_linear = torch.nn.Linear(
                entity_dim + relation_dim, layer_dim, bias=True
            )
            self.po_linear = torch.nn.Linear(
                relation_dim + entity_dim, layer_dim, bias=True
            )
        elif self.g_func == "bilinear":
            self.sp_bilinear = torch.nn.Bilinear(
                entity_dim, relation_dim, entity_dim, bias=True
            )
            self.po_bilinear = torch.nn.Bilinear(
                relation_dim, entity_dim, entity_dim, bias=True
            )
        else:
            raise ValueError

    def score_emb(
        self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str
    ) -> Tensor:
        n = p_emb.size(0)
        if combine == "spo":
            if self.g_func == "linear":
                sp_embed = torch.cat((s_emb, p_emb), 1)
                sp_embed = self.dropout(sp_embed)
                po_embed = torch.cat((p_emb, o_emb), 1)
                po_embed = self.dropout(po_embed)
                sp_embed = self.sp_linear(sp_embed)
                po_embed = self.po_linear(po_embed)
            elif self.g_func == "bilinear":
                sp_embed = self.sp_bilinear(s_emb, p_emb)
                po_embed = self.sp_bilinear(p_emb, o_emb)
            else:
                raise ValueError
            out = torch.einsum(
                "ij,ij->i", sp_embed, po_embed
            )  # compute row-wise dot product
        elif combine == "sp_":
            score_list = []
            for o_emb_single in o_emb:
                o_emb_rep = o_emb_single.repeat((n, 1))
                if self.g_func == "linear":
                    sp_embed = torch.cat((s_emb, p_emb), 1)
                    sp_embed = self.dropout(sp_embed)
                    po_embed = torch.cat((p_emb, o_emb_rep), 1)
                    po_embed = self.dropout(po_embed)
                    sp_embed = self.sp_linear(sp_embed)
                    po_embed = self.po_linear(po_embed)
                elif self.g_func == "bilinear":
                    sp_embed = self.sp_bilinear(s_emb, p_emb)
                    po_embed = self.sp_bilinear(p_emb, o_emb_rep)
                else:
                    raise ValueError
                embed_score = torch.einsum("ij,ij->i", sp_embed, po_embed)
                score_list.append(embed_score)
            out = torch.stack(score_list, 1)
        elif combine == "_po":
            score_list = []
            for s_emb_single in s_emb:
                s_emb_rep = s_emb_single.repeat((n, 1))
                if self.g_func == "linear":
                    sp_embed = torch.cat((s_emb_rep, p_emb), 1)
                    sp_embed = self.dropout(sp_embed)
                    po_embed = torch.cat((p_emb, o_emb), 1)
                    po_embed = self.dropout(po_embed)
                    sp_embed = self.sp_linear(sp_embed)
                    po_embed = self.po_linear(po_embed)
                elif self.g_func == "bilinear":
                    sp_embed = self.sp_bilinear(s_emb_rep, p_emb)
                    po_embed = self.sp_bilinear(p_emb, o_emb)
                else:
                    raise ValueError
                embed_score = torch.einsum("ij,ij->i", sp_embed, po_embed)
                score_list.append(embed_score)
            out = torch.stack(score_list, 1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class SemanticMatchingEnergy(KgeModel):
    r"""Implementation of the Semantic Matching Energy KGE model."""

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
