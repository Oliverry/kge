import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.aggregation import AutoencoderReduction, PcaReduction, Concatenation, OneToN
from kge.model.ensemble.embedding_evaluator import KgeAdapter, FineTuning
from kge.model.ensemble.load_pretrain import fetch_embedding


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
            self.aggregation = Concatenation(config, self.configuration_key)
        elif aggregation_option == "pca":
            self.aggregation = PcaReduction(config, self.configuration_key)
        elif aggregation_option == "autoencoder":
            self.aggregation = AutoencoderReduction(config, self.configuration_key)
        elif aggregation_option == "oneton":
            self.aggregation = OneToN(dataset, config, self.configuration_key)
        else:
            raise Exception("Unknown dimensionality reduction: " + aggregation_option)

        # Lookup and initiate evaluator method
        if evaluator_option == "kge_adapter":
            self.evaluator = KgeAdapter(dataset, config, self.configuration_key)
        elif evaluator_option == "finetuning":
            self.evaluator = FineTuning(dataset, config, self.configuration_key)
        else:
            raise Exception("Unknown evaluator: "+evaluator_option)

        # Start training of dimensionality reduction method
        if config.get("job.type") == "train":
            self.aggregation.train_aggregation(self.submodels)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.aggregation.aggregate_entities(s_emb, s)
        p_emb = self.aggregation.aggregate_relations(p_emb, p)
        o_emb = self.aggregation.aggregate_entities(o_emb, o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "spo")
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.aggregation.aggregate_entities(s_emb, s)
        p_emb = self.aggregation.aggregate_relations(p_emb, p)
        o_emb = self.aggregation.aggregate_entities(o_emb, o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "sp_")
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.aggregation.aggregate_entities(s_emb, s)
        p_emb = self.aggregation.aggregate_relations(p_emb, p)
        o_emb = self.aggregation.aggregate_entities(o_emb, o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "_po")
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.aggregation.aggregate_entities(s_emb, s)
        p_emb = self.aggregation.aggregate_relations(p_emb, p)
        o_emb = self.aggregation.aggregate_entities(o_emb, o)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "s_o")
        return scores

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        # fetch and reduce model embeddings
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.aggregation.aggregate_entities(s_emb, s)
        p_emb = self.aggregation.aggregate_relations(p_emb, p)
        o_emb = self.aggregation.aggregate_entities(o_emb, o)

        # fetch and reduce additional entity subset
        sub_emb_list = []
        obj_emb_list = []
        for model in self.submodels:
            model_sub_emb = fetch_embedding(model, "s", entity_subset)
            model_obj_emb = fetch_embedding(model, "o", entity_subset)
            sub_emb_list.append(model_sub_emb)
            obj_emb_list.append(model_obj_emb)
        sub_emb = torch.cat(sub_emb_list, dim=1)
        obj_emb = torch.cat(obj_emb_list, dim=1)
        sub_emb = self.aggregation.aggregate_entities(sub_emb)
        obj_emb = self.aggregation.aggregate_entities(obj_emb)

        sp_scores = self.evaluator.score_emb(s_emb, p_emb, obj_emb, "sp_")
        po_scores = self.evaluator.score_emb(sub_emb, p_emb, o_emb, "_po")

        res = torch.cat((sp_scores, po_scores), dim=1)
        return res

    def fetch_model_embeddings(self, s: Tensor = None, p: Tensor = None, o: Tensor = None) -> (Tensor, Tensor, Tensor):
        s_emb_list = []
        p_emb_list = []
        o_emb_list = []
        for model in self.submodels:
            model_s_emb = fetch_embedding(model, "s", s)
            model_p_emb = fetch_embedding(model, "p", p)
            model_o_emb = fetch_embedding(model, "o", o)
            s_emb_list.append(model_s_emb)
            p_emb_list.append(model_p_emb)
            o_emb_list.append(model_o_emb)
        s_embeds = torch.cat(s_emb_list, dim=1)
        p_embeds = torch.cat(p_emb_list, dim=1)
        o_embeds = torch.cat(o_emb_list, dim=1)
        return s_embeds, p_embeds, o_embeds
