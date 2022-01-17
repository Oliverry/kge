from typing import List

import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.dim_reduction import AutoencoderReduction, ConcatenationReduction
from kge.model.ensemble.embedding_evaluator import KgeAdapter, FineTuning
from kge.model.kge_model import KgeBase


class EmbeddingEnsemble(Ensemble):

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
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        dim_reduction_str = self.get_option("dim_reduction")
        evaluator_str = self.get_option("evaluator")
        if dim_reduction_str == "concat":
            self.dim_reduction = ConcatenationReduction(config, "concat")
        elif dim_reduction_str == "autoencoder":
            self.dim_reduction = AutoencoderReduction(config, "autoencoder")

        if evaluator_str == "kge_adapter":
            self.evaluator = KgeAdapter(dataset, config, "kge_adapter")
        elif evaluator_str == "finetuning":
            self.evaluator = FineTuning(dataset, config, "finetuning")

        if config.get("job.type") == "train":
            self.dim_reduction.train_dim_reduction(self.submodels)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        n = s.size()[0]
        s_embeds = None
        p_embeds = None
        o_embeds = None
        for idx, model in enumerate(self.submodels):
            model_s_embeds = model.get_s_embedder().embed(s).detach()
            model_s_embeds = model_s_embeds.view(n, 1, -1)
            model_p_embeds = model.get_p_embedder().embed(p).detach()
            model_p_embeds = model_p_embeds.view(n, 1, -1)
            model_o_embeds = model.get_o_embedder().embed(o).detach()
            model_o_embeds = model_o_embeds.view(n, 1, -1)
            if s_embeds is None:
                s_embeds = model_s_embeds
                p_embeds = model_p_embeds
                o_embeds = model_o_embeds
            else:
                s_embeds = torch.cat((s_embeds, model_s_embeds), 1)
                p_embeds = torch.cat((p_embeds, model_p_embeds), 1)
                o_embeds = torch.cat((o_embeds, model_o_embeds), 1)
        s_embeds = self.dim_reduction.reduce_entities(s_embeds)
        p_embeds = self.dim_reduction.reduce_relations(p_embeds)
        o_embeds = self.dim_reduction.reduce_entities(o_embeds)
        scores = self.evaluator.score_emb(s_embeds, p_embeds, o_embeds, "spo")
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        # not implemented
        scores = []
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp(s, p)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            model_scores = torch.transpose(model_scores, 0, 1)
            scores.append(model_scores)
        for idx in range(0, len(self.submodels)):
            pass
        return self.evaluator(scores)

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        # not implemented
        scores = None
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_po(p, o)
            model_scores = torch.unsqueeze(model_scores, dim=0)
            if scores is None:
                scores = model_scores
            else:
                scores = torch.cat((scores, model_scores), 0)
        return self.evaluator(scores)

    def score_sp_po(
            self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        # embed s, p, o
        n = s.size()[0]
        s_embeds = None
        p_embeds = None
        o_embeds = None
        for idx, model in enumerate(self.submodels):
            model_s_embeds = model.get_s_embedder().embed(s)
            model_s_embeds = model_s_embeds.view(n, 1, -1)
            model_p_embeds = model.get_p_embedder().embed(p)
            model_p_embeds = model_p_embeds.view(n, 1, -1)
            model_o_embeds = model.get_o_embedder().embed(o)
            model_o_embeds = model_o_embeds.view(n, 1, -1)
            if s_embeds is None:
                s_embeds = model_s_embeds
                p_embeds = model_p_embeds
                o_embeds = model_o_embeds
            else:
                s_embeds = torch.cat((s_embeds, model_s_embeds), 1)
                p_embeds = torch.cat((p_embeds, model_p_embeds), 1)
                o_embeds = torch.cat((o_embeds, model_o_embeds), 1)

        # embed entity subset, assuming s embedder and o embedder are the same
        entity_subset_embeds = None
        for idx, model in enumerate(self.submodels):
            if entity_subset is not None:
                model_s_embeds = model.get_s_embedder().embed(entity_subset)
            else:
                model_s_embeds = model.get_s_embedder().embed_all()
            m = model_s_embeds.size()[0]
            model_s_embeds = model_s_embeds.view(m, 1, -1)
            if entity_subset_embeds is None:
                entity_subset_embeds = model_s_embeds
            else:
                entity_subset_embeds = torch.cat((entity_subset_embeds, model_s_embeds), 1)
        # dim reduction
        s_embeds = self.dim_reduction.reduce_entities(s_embeds)
        p_embeds = self.dim_reduction.reduce_relations(p_embeds)
        o_embeds = self.dim_reduction.reduce_entities(o_embeds)
        entity_subset_embeds = self.dim_reduction.reduce_entities(entity_subset_embeds)

        # applying evaluator
        sp_scores = self.evaluator.score_emb(s_embeds, p_embeds, entity_subset_embeds, "sp_")
        po_scores = self.evaluator.score_emb(entity_subset_embeds, p_embeds, o_embeds, "_po")

        res = torch.cat((sp_scores, po_scores), dim=1)
        return res

