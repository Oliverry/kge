import torch
from torch import nn, Tensor

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


class DklrScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb_with_descriptions(self, s_emb, p_emb, o_emb, s_desc, o_desc, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
            out += -F.pairwise_distance(s_desc + p_emb, o_desc, p=self._norm)
            out += -F.pairwise_distance(s_desc + p_emb, o_emb, p=self._norm)
            out += -F.pairwise_distance(s_emb + p_emb, o_desc, p=self._norm)
        elif combine == "sp_":
            out = -torch.cdist(s_emb + p_emb, o_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(s_desc + p_emb, o_desc, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(s_desc + p_emb, o_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(s_emb + p_emb, o_desc, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
        elif combine == "_po":
            out = -torch.cdist(o_emb - p_emb, s_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(o_desc - p_emb, s_desc, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(o_desc - p_emb, s_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
            out += -torch.cdist(o_emb - p_emb, s_desc, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist", )
        else:
            # TODO
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class DKLR(KgeModel):
    r"""Implementation of the TransE KGE model."""

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
            scorer=DklrScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        # create descriptions embeddings
        num_entities = self.config.get("dataset.num_entities")
        embed_dim = self.get_option("entity_embedder.dim")
        self.desc_embeddings = torch.nn.Embedding(num_entities, embed_dim)

        # encode descriptions to embeddings if train model
        if not init_for_load_only:
            self.pretrain_embeddings()

    def pretrain_embeddings(self):
        pass

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)

        s_desc = self.desc_embeddings(s)
        o_desc = self.desc_embeddings(o)

        return self._scorer.score_emb_with_descriptions(s_emb, p_emb, o_emb, s_desc, o_desc, combine="spo").view(-1)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        s_desc = self.desc_embeddings(s)
        o_desc = self.desc_embeddings(o)

        return self._scorer.score_emb_with_descriptions(s_emb, p_emb, o_emb, s_desc, o_desc, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        p_emb = self.get_p_embedder().embed(p)

        s_desc = self.desc_embeddings(s)
        o_desc = self.desc_embeddings(o)

        return self._scorer.score_emb_with_descriptions(s_emb, p_emb, o_emb, s_desc, o_desc, combine="_po")

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        if p is None:
            p_emb = self.get_p_embedder().embed_all()
        else:
            p_emb = self.get_p_embedder().embed(p)

        s_desc = self.desc_embeddings(s)
        o_desc = self.desc_embeddings(o)

        return self._scorer.score_emb_with_descriptions(s_emb, p_emb, o_emb, s_desc, o_desc, combine="s_o")

    def score_sp_po(
            self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)

        s_desc = self.desc_embeddings(s)
        o_desc = self.desc_embeddings(o)

        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
                all_entities_desc = self.desc_embeddings(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
                all_entities_idx = torch.arange(
                    self.config.get("dataset.num_entities"), dtype=torch.long, device=self.config.get("job.device")
                )
                all_entities_desc = self.desc_embeddings(all_entities_idx)

            sp_scores = self._scorer.score_emb_with_descriptions(
                s_emb, p_emb, all_entities, s_desc, all_entities_desc, combine="sp_"
            )
            po_scores = self._scorer.score_emb_with_descriptions(
                all_entities, p_emb, o_emb, all_entities_desc, o_desc, combine="_po"
            )
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
                all_entities_desc = self.desc_embeddings(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
                all_entities_idx = torch.arange(
                    self.config.get("dataset.num_entities"), dtype=torch.long, device=self.config.get("job.device")
                )
                all_entities_desc = self.desc_embeddings(all_entities_idx)
            sp_scores = self._scorer.score_emb_with_descriptions(
                s_emb, p_emb, all_objects, s_desc, all_entities_desc, combine="sp_"
            )
            po_scores = self._scorer.score_emb_with_descriptions(
                all_subjects, p_emb, o_emb, all_entities_desc, o_desc, combine="_po"
            )
        return torch.cat((sp_scores, po_scores), dim=1)
