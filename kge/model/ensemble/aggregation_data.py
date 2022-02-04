import torch
from torch import Tensor
from torch.utils.data import Dataset

from kge.model import KgeModel


def fetch_multiple_embeddings(models, target, indexes: Tensor = None) -> Tensor:
    """
    Return tensor of size n times m times dim, where n is the length of the index tensor,
    m is the number of models and dim is the embedding length.
    :param models:
    :param target:
    :param indexes:
    :return:
    """
    emb_list = []
    for model in models:
        model_emb = fetch_embedding(model, target, indexes)
        emb_list.append(model_emb)
    embeds = torch.cat(emb_list, dim=1)
    return embeds


def fetch_embedding(model: KgeModel, target, idxs: Tensor = None) -> Tensor:
    """
    Fetches the embedding of a given model and indexes using the specified embedder
    :param target:
    :param model: The specified model
    :param idxs: Given indexes
    :return: Tensor of embeddings with shape n x 1 x E
    """
    if target == "s":
        if idxs is None:
            out = model.get_s_embedder().embed_all()
        else:
            out = model.get_s_embedder().embed(idxs)
    elif target == "o":
        if idxs is None:
            out = model.get_o_embedder().embed_all()
        else:
            out = model.get_o_embedder().embed(idxs)
    elif target == "p":
        if idxs is None:
            out = model.get_p_embedder().embed_all()
        else:
            out = model.get_p_embedder().embed(idxs)
    else:
        raise Exception("Unknown target embedder is specified: " + target)
    n = out.size()[0]
    out = out.view(n, 1, -1)
    return out.detach()


class AggregationDataset(Dataset):
    def __init__(self, models, target="entity"):
        """
        Creates a new dataset for unsupervised learning of aggregation models.
        The data has the format n times m times dim_m
        :param models:
        :param mode: either "entity" or "relation"
        """
        self.data = None
        for idx, model in enumerate(models):
            if target == "entity":
                if model.get_s_embedder() == model.get_o_embedder():
                    m_embeds = fetch_embedding(model, "s")
                else:
                    raise Exception("Different embedders are not supported.")
            elif target == "relation":
                m_embeds = fetch_embedding(model, "p")
            else:
                raise ValueError("Unknown target for dataset creation.")
            n = m_embeds.size()[0]
            m_embeds = m_embeds.view(n, 1, -1)
            if self.data is None:
                self.data = m_embeds
            elif self.data is not None:
                self.data = torch.cat((self.data, m_embeds), 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]