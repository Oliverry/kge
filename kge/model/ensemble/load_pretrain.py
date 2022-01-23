from torch import Tensor

from kge.model import KgeModel


def fetch_embedding(model: KgeModel, embedder, idxs: Tensor = None) -> Tensor:
    """
    Fetches the embedding of a given model and indexes using the specified embedder
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
        raise Exception("embedder has to be specified.")  # TODO specify exception
    n = out.size()[0]
    out = out.view(n, 1, -1)
    return out.detach()
