import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.aggregation import AutoencoderReduction, PcaReduction, Concatenation, OneToN
from kge.model.ensemble.embedding_evaluator import KgeAdapter, FineTuning


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
        # Lookup and initiate dimensionality reduction method
        aggregation_option = self.get_option("aggregation")
        evaluator_option = self.get_option("evaluator")
        if aggregation_option == "concat":
            self.aggregation = Concatenation(self.submodels, config, self.configuration_key)
        elif aggregation_option == "pca":
            self.aggregation = PcaReduction(self.submodels, config, self.configuration_key)
        elif aggregation_option == "autoencoder":
            self.aggregation = AutoencoderReduction(self.submodels, config, self.configuration_key)
        elif aggregation_option == "oneton":
            self.aggregation = OneToN(self.submodels, dataset, config, self.configuration_key)
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
            self.aggregation.train_aggregation(self.submodels)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.aggregation.aggregate("s", s)
        p_emb = self.aggregation.aggregate("p", p)
        o_emb = self.aggregation.aggregate("o", o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "spo")
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate("s", s)
        p_emb = self.aggregation.aggregate("p", p)
        o_emb = self.aggregation.aggregate("o", o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "sp_")
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate("s", s)
        p_emb = self.aggregation.aggregate("p", p)
        o_emb = self.aggregation.aggregate("o", o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "_po")
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb = self.aggregation.aggregate("s", s)
        p_emb = self.aggregation.aggregate("p", p)
        o_emb = self.aggregation.aggregate("o", o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "s_o")
        return scores

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        # aggregate standard model embeddings
        s_emb = self.aggregation.aggregate("s", s)
        p_emb = self.aggregation.aggregate("p", p)
        o_emb = self.aggregation.aggregate("o", o)

        # aggregate additional entity subset
        sub_emb = self.aggregation.aggregate("s", entity_subset)
        obj_emb = self.aggregation.aggregate("o", entity_subset)

        sp_scores = self.evaluator.score_emb(s_emb, p_emb, obj_emb, "sp_")
        po_scores = self.evaluator.score_emb(sub_emb, p_emb, o_emb, "_po")

        res = torch.cat((sp_scores, po_scores), dim=1)
        return res
