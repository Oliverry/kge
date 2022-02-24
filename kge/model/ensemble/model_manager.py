from typing import List

import torch
from torch import Tensor

from kge.model import KgeModel, ReciprocalRelationsModel, RotatE, ConvE


def fetch_embedding(model: KgeModel, target, indexes: Tensor = None) -> Tensor:
    """
    Fetches the embedding of a given model and indexes using the specified embedder
    :param target:
    :param model: The specified model
    :param indexes: Given indexes
    :return: Tensor of embeddings with shape n x 1 x E
    """
    # check if certain type of model is used
    is_rrm = contains_model(model, ReciprocalRelationsModel)
    is_rotate = contains_model(model, RotatE)

    if target == "p" and is_rrm:
        # check if reciprocal relations model is used and split relation embedding
        if indexes is None:
            out_rrm = model.get_p_embedder().embed_all()
            half_t = int(out_rrm.size()[0]/2)
            out_one = out_rrm[:half_t]
            out_two = out_rrm[half_t:]
        else:
            out_one = model.get_p_embedder().embed(indexes)
            out_two = model.get_p_embedder().embed(indexes + model.dataset.num_relations())
        out = 0.5 * (out_one + out_two)
    else:
        # normal embedding fetching
        if target == "s":
            embedder = model.get_s_embedder()
        elif target == "p":
            embedder = model.get_p_embedder()
        elif target == "o":
            embedder = model.get_o_embedder()
        else:
            raise Exception("Unknown target embedder is specified: " + target)

        if indexes is None:
            out = embedder.embed_all()
        else:
            out = embedder.embed(indexes)

    # check if rotate model is used and convert relation embedding
    if target == "p" and is_rotate:
        re = torch.cos(out)
        img = torch.sin(out)
        out = torch.cat((re, img), dim=1)

    return out.detach()


def contains_model(model: KgeModel, model_type):
    contained = (isinstance(model, ReciprocalRelationsModel) and isinstance(model.base_model(), model_type))
    return contained or isinstance(model, model_type)


class ModelManager:

    def __init__(self, models: List[KgeModel]):
        self.models = models

        # lookup model specific entity and relation sizes
        self.dims = {}
        for idx, model in enumerate(self.models):
            entity_emb = fetch_embedding(model, "s", torch.Tensor(0))
            relation_emb = fetch_embedding(model, "p", torch.Tensor(0))
            entity_dim = entity_emb.size()[1]
            relation_dim = relation_emb.size()[1]
            self.dims[idx] = (entity_dim, relation_dim)

    def num_models(self):
        return len(self.models)

    def model_embed_dims(self):
        """
        Return a dictionary with the index of a model as key and corresponding value as (entity_dim, relation_dim).
        :return:
        """
        return self.dims

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        scores_list = []
        for model in self.models:
            model_scores = model.score_spo(s, p, o, direction).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=1)
        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.models:
            model_scores = model.score_sp(s, p, o).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        return scores

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.models:
            model_scores = model.score_po(p, o, s).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        return scores

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        scores_list = []
        for model in self.models:
            model_scores = model.score_so(s, o, p).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        return scores

    def score_sp_po(
            self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        scores_list = []
        for model in self.models:
            model_scores = model.score_sp_po(s, p, o, entity_subset).detach()
            scores_list.append(model_scores)
        scores = torch.stack(scores_list, dim=2)
        return scores

    def fetch_model_embeddings(self, target, indexes: Tensor = None) -> Tensor:
        """
        Return tensor of size n times m times dim, where n is the length of the index tensor,
        m is the number of models and dim is the embedding length.
        :param target:
        :param indexes:
        :return:
        """
        embeds = {}
        for idx, model in enumerate(self.models):
            model_emb = fetch_embedding(model, target, indexes)
            embeds[idx] = model_emb
        return embeds
