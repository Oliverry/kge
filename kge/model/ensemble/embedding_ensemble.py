import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.dim_reduction import AutoencoderReduction


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
        if dim_reduction_str == "autoencoder":
            self.dim_reduction = AutoencoderReduction(config, "autoencoder")

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
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
        s_embeds = self.dim_reduction.reduce_entities(s_embeds)
        p_embeds = self.dim_reduction.reduce_relations(p_embeds)
        o_embeds = self.dim_reduction.reduce_entities(o_embeds)
        print("done")

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
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
        col_size = 2*entity_subset.size()[0]
        col_scores = [torch.empty(len(self.submodels), s.size()[0]) for _ in range(0, col_size)]
        res = torch.empty(col_size, s.size()[0])
        for idx, model in enumerate(self.submodels):
            model_scores = model.score_sp_po(s, p, o, entity_subset)
            model_scores = torch.transpose(model_scores, 0, 1)
            for col_idx in range(0, col_size):
                col_scores[col_idx][idx] = model_scores[idx]
        for idx in range(0, col_size):
            t = torch.transpose(col_scores[idx], 0, 1)
            col_res = self.evaluator(t)
            res[idx] = col_res
        res = torch.transpose(res, 0, 1)
        return res

    def save(self):
        pass

    def load(self, savepoint):
        pass
