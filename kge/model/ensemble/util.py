from enum import Enum

from kge.model import KgeModel, ReciprocalRelationsModel


def contains_model(model: KgeModel, model_type):
    """
    Checks if the given model contains the specified model type.
    :param model: The KGE model to be checked.
    :param model_type: The model type to be checked.
    :return: Boolean whether the check is true or not.
    """
    contained = isinstance(model, ReciprocalRelationsModel) and isinstance(
        model.base_model(), model_type
    )
    return contained or isinstance(model, model_type)


class EmbeddingTarget(Enum):
    """
    Enumeration to check the embedding target, where we distinguish between subject, predicate and object.
    """

    Subject = (1,)
    Predicate = (2,)
    Object = 3


class EmbeddingType(Enum):
    """
    Enumeration to check the embedding type, where we distinguish between entity and relation.
    """

    Entity = (1,)
    Relation = 2
