from typing import List, Dict

import torch
from torch import Tensor

from kge import Config
from kge.model import KgeModel, ReciprocalRelationsModel, RotatE
from kge.model.ensemble.util import EmbeddingTarget, contains_model


def fetch_embedding(
    model: KgeModel, target: EmbeddingTarget, indexes: Tensor = None
) -> Tensor:
    """
    Fetches the embedding of a given KGE model and an embedding target.
    If indexes is given, entities or relations embeddings for index IDs are fetched, else all entities or relation
    embeddings are fetched.
    Fetches the embedding of a given model and indexes using the specified embedder.
    Returns a tensor of n times E, where n is the number of indexes or number of entities and relations.
    E is the embedding size of the KGE model.
    :param target: Subject, predicate or object target of fetching.
    :param model: The KGE model for fetching.
    :param indexes: IDs of entities or relations to be fetched.
    :return: Tensor of embeddings.
    """
    # check if certain type of model is used
    is_rrm = contains_model(model, ReciprocalRelationsModel)
    is_rotate = contains_model(model, RotatE)

    # fetch model embeddings
    if target == EmbeddingTarget.Predicate and is_rrm:
        # check if reciprocal relations model is used and process relation embedding
        if indexes is None:
            out_rrm = model.get_p_embedder().embed_all()
            half_t = int(out_rrm.size()[0] / 2)
            out_one = out_rrm[:half_t]
            out_two = out_rrm[half_t:]
        else:
            out_one = model.get_p_embedder().embed(indexes)
            out_two = model.get_p_embedder().embed(
                indexes + model.dataset.num_relations()
            )
        out = 0.5 * (out_one + out_two)
    else:
        # normal embedding fetching
        if target == EmbeddingTarget.Subject:
            embedder = model.get_s_embedder()
        elif target == EmbeddingTarget.Predicate:
            embedder = model.get_p_embedder()
        elif target == EmbeddingTarget.Object:
            embedder = model.get_o_embedder()
        else:
            raise ValueError("Unknown target embedder is specified: " + str(target))

        if indexes is None:
            out = embedder.embed_all()
        else:
            out = embedder.embed(indexes)

    # postprocess predicate embeddings if model is RotatE
    if target == EmbeddingTarget.Predicate and is_rotate:
        re = torch.cos(out)
        img = torch.sin(out)
        out = torch.cat((re, img), dim=1)

    return out.detach()


class ModelManager:
    """
    Class to manage the base models and handle queries for scores and embeddings.
    """

    def __init__(self, config: Config, models: List[KgeModel]):
        self.models = models

        # lookup model specific dimension sizes
        # use entity embedding, as relations are preprocessed to the same size
        self.dims = {}
        for idx, model in enumerate(self.models):
            entity_emb = fetch_embedding(
                model,
                EmbeddingTarget.Subject,
                torch.tensor([0], device=config.get("job.device")),
            )
            entity_dim = entity_emb.size()[1]
            self.dims[idx] = entity_dim

    def num_models(self):
        """
        Return the number of base models in the given model manager.
        :return: Number of base models.
        """
        return len(self.models)

    def get_model_dims(self):
        """
        Returns dictionary of model-specific embedding dimensions.
        Given a model index, the entity and relation embedding size is returned.
        :return: Dictionary of embedding sizes.
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

    def fetch_model_embeddings(
        self, target: EmbeddingTarget, indexes: Tensor = None
    ) -> Dict[int, Tensor]:
        """
        Fetches the embeddings of all base models, given an embedding target.
        If indexes are given, they are used as entity or relation IDs, otherwise all entities and relations are fetched.
        Returns a dictionary, where given a model index, the embeddings are delivered.
        The tensors have the form n times E, where n is the length of indexes or number of all entities or relations.
        E is the model specific embedding length.
        :param target: Subject, predicate or object target for fetching.
        :param indexes: IDs of entities or relations.
        :return: Dictionary of model specific embeddings given the model index.
        """
        embeds = {}
        for model_idx, model in enumerate(self.models):
            model_emb = fetch_embedding(model, target, indexes)
            embeds[model_idx] = model_emb
        return embeds
