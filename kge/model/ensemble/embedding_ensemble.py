from typing import List

import torch
import torch.nn.functional
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.aggregation import AutoencoderReduction, Concatenation, OneToN, PcaReduction, MeanReduction
from kge.model.ensemble.embedding_evaluator import KgeAdapter, FineTuning
from kge.model.ensemble.model_manager import EmbeddingTarget


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

        # Lookup  embedding ensemble options
        self.normalize_p = self.get_option("normalize_p")

        # Lookup and initiate dimensionality reduction method
        aggregation_option = self.get_option("aggregation")
        evaluator_option = self.get_option("evaluator")
        if aggregation_option == "concat":
            self.aggregation = Concatenation(self.model_manager, config, self.configuration_key)
        elif aggregation_option == "mean":
            self.aggregation = MeanReduction(self.model_manager, config, self.configuration_key)
        elif aggregation_option == "pca":
            self.aggregation = PcaReduction(self.model_manager, dataset, config, self.configuration_key)
        elif aggregation_option == "autoencoder":
            self.aggregation = AutoencoderReduction(self.model_manager, config, self.configuration_key)
        elif aggregation_option == "oneton":
            self.aggregation = OneToN(self.model_manager, dataset, config, self.configuration_key)
        else:
            raise Exception("Unknown dimensionality reduction: " + aggregation_option)

        # Lookup and initiate evaluator method
        if evaluator_option == "kge_adapter":
            self.evaluator = KgeAdapter(dataset, config, self.configuration_key)
        elif evaluator_option == "finetuning":
            self.evaluator = FineTuning(dataset, config, self.configuration_key)
        else:
            raise Exception("Unknown evaluator: " + evaluator_option)

        # Start training of dimensionality reduction method
        if config.get("job.type") == "train":
            self.aggregation.train_aggregation()

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, s)
        p_emb = self.aggregation.aggregate(EmbeddingTarget.Predicate, p)
        o_emb = self.aggregation.aggregate(EmbeddingTarget.Object, o)
        s_emb = self._postprocess(s_emb)
        p_emb = self._postprocess(p_emb)
        o_emb = self._postprocess(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "spo")
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, s)
        p_emb = self.aggregation.aggregate(EmbeddingTarget.Predicate, p)
        o_emb = self.aggregation.aggregate(EmbeddingTarget.Object, o)
        s_emb = self._postprocess(s_emb)
        p_emb = self._postprocess(p_emb)
        o_emb = self._postprocess(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "sp_")
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, s)
        p_emb = self.aggregation.aggregate(EmbeddingTarget.Predicate, p)
        o_emb = self.aggregation.aggregate(EmbeddingTarget.Object, o)
        s_emb = self._postprocess(s_emb)
        p_emb = self._postprocess(p_emb)
        o_emb = self._postprocess(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "_po")
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, s)
        p_emb = self.aggregation.aggregate(EmbeddingTarget.Predicate, p)
        o_emb = self.aggregation.aggregate(EmbeddingTarget.Object, o)
        s_emb = self._postprocess(s_emb)
        p_emb = self._postprocess(p_emb)
        o_emb = self._postprocess(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "s_o")
        return scores

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        # aggregate standard model embeddings
        s_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, s)
        p_emb = self.aggregation.aggregate(EmbeddingTarget.Predicate, p)
        o_emb = self.aggregation.aggregate(EmbeddingTarget.Object, o)
        s_emb = self._postprocess(s_emb)
        p_emb = self._postprocess(p_emb)
        o_emb = self._postprocess(o_emb)

        # aggregate additional entity subset
        sub_emb = self.aggregation.aggregate(EmbeddingTarget.Subject, entity_subset)
        obj_emb = self.aggregation.aggregate(EmbeddingTarget.Object, entity_subset)
        sub_emb = self._postprocess(sub_emb)
        obj_emb = self._postprocess(obj_emb)

        sp_scores = self.evaluator.score_emb(s_emb, p_emb, obj_emb, "sp_")
        po_scores = self.evaluator.score_emb(sub_emb, p_emb, o_emb, "_po")

        res = torch.cat((sp_scores, po_scores), dim=1)
        return res

    def _postprocess(self, embed: Tensor) -> Tensor:
        if self.normalize_p > 0:
            with torch.no_grad():
                embed = torch.nn.functional.normalize(embed, p=self.normalize_p, dim=1)
        return embed

    def penalty(self, **kwargs) -> List[Tensor]:
        result = super().penalty(**kwargs)
        result += self.aggregation.penalty(**kwargs)
        return result

