from typing import List

import torch
from torch import Tensor

from kge import Config, Dataset
from kge.model import Ensemble
from kge.model.ensemble.dim_reduction import AutoencoderReduction, ConcatenationReduction
from kge.model.ensemble.embedding_evaluator import KgeAdapter, FineTuning
from kge.model.kge_model import KgeModel


# TODO maybe do code optimization
def fetch_embedding(model: KgeModel, embedder, idxs: Tensor = None) -> Tensor:
    """
    Fetches the embedding of a given model and indexes using the specified embedder.
    :param model: The specified model
    :param idxs: Given indexes
    :param embedder: Can be "s", "p", "o"
    :return: Tensor of embeddings with shape n x 1 x E
    """
    if embedder == "s":
        if idxs is None:
            out = model.get_s_embedder().embed_all()
        else:
            out = model.get_s_embedder().embed(idxs)
    elif embedder == "o":
        if idxs is None:
            out = model.get_o_embedder().embed_all()
        else:
            out = model.get_o_embedder().embed(idxs)
    elif embedder == "p":
        if idxs is None:
            out = model.get_p_embedder().embed_all()
        else:
            out = model.get_p_embedder().embed(idxs)
    else:
        raise Exception("embedder has to be specified.")  # TODO specifiy exception
    n = out.size()[0]
    out = out.view(n, 1, -1)
    return out.detach()


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
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.dim_reduction.reduce_entities(s_emb)
        p_emb = self.dim_reduction.reduce_relations(p_emb)
        o_emb = self.dim_reduction.reduce_entities(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "spo")
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.dim_reduction.reduce_entities(s_emb)
        p_emb = self.dim_reduction.reduce_relations(p_emb)
        o_emb = self.dim_reduction.reduce_entities(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "sp_")
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.dim_reduction.reduce_entities(s_emb)
        p_emb = self.dim_reduction.reduce_relations(p_emb)
        o_emb = self.dim_reduction.reduce_entities(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "_po")
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.dim_reduction.reduce_entities(s_emb)
        p_emb = self.dim_reduction.reduce_relations(p_emb)
        o_emb = self.dim_reduction.reduce_entities(o_emb)
        scores = self.evaluator.score_emb(s_emb, p_emb, o_emb, "s_o")
        return scores

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        # fetch and reduce model embeddings
        s_emb, p_emb, o_emb = self.fetch_model_embeddings(s, p, o)
        s_emb = self.dim_reduction.reduce_entities(s_emb)
        p_emb = self.dim_reduction.reduce_relations(p_emb)
        o_emb = self.dim_reduction.reduce_entities(o_emb)

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
        sub_emb = self.dim_reduction.reduce_entities(sub_emb)
        obj_emb = self.dim_reduction.reduce_entities(obj_emb)

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
