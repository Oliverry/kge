import torch
from torch import Tensor
from torch.utils.data import Dataset

from kge.model import KgeModel, ReciprocalRelationsModel, RotatE, ConvE


def fetch_model_embeddings(models, target, indexes: Tensor = None) -> Tensor:
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


def fetch_embedding(model: KgeModel, target, indexes: Tensor = None) -> Tensor:
    """
    Fetches the embedding of a given model and indexes using the specified embedder
    :param target:
    :param model: The specified model
    :param indexes: Given indexes
    :return: Tensor of embeddings with shape n x 1 x E
    """
    # check if certain type of model is used
    is_rotate = isinstance(model, RotatE) or (isinstance(model, ReciprocalRelationsModel) and
                                              isinstance(model.base_model(), RotatE))
    is_conve = isinstance(model, ConvE) or (isinstance(model, ReciprocalRelationsModel) and
                                            isinstance(model.base_model(), ConvE))

    # check if reciprocal relations model is used and split relation embedding
    if target == "p" and isinstance(model, ReciprocalRelationsModel):
        if indexes is None:
            out_rrm = model.get_p_embedder().embed_all()
            out_one, out_two = torch.tensor_split(out_rrm, 2)
        else:
            out_one = model.get_p_embedder().embed(indexes)
            out_two = model.get_p_embedder().embed(indexes + model.dataset.num_relations())
        n = out_one.size()[0]
        out_one = out_one.view(n, 1, -1)
        out_two = out_two.view(n, 1, -1)
        out = torch.cat((out_one, out_two), dim=1)
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
        n = out.size()[0]
        out = out.view(n, 1, -1)

    # check if conve model and remove bias term hack
    if is_conve:
        out_sz = out.size()
        emb_dim = out_sz[len(out_sz) - 1]
        out = out[..., :emb_dim - 1]

    # check if rotate model is used and convert relation embedding
    if target == "p" and is_rotate:
        re = torch.cos(out)
        img = torch.sin(out)
        out = torch.cat((re, img), dim=2)
    return out.detach()


class AggregationDataset(Dataset):
    def __init__(self, data):
        """
        Creates a new dataset for unsupervised learning of aggregation models.
        The data has the format n times m times dim_m
        :param models:
        :param target: either "entity" or "relation"
        """
        # TODO what if subject, object embedder are different?
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]
