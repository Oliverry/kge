from enum import Enum

from kge.model import KgeModel, ReciprocalRelationsModel


def contains_model(model: KgeModel, model_type):
    contained = (isinstance(model, ReciprocalRelationsModel) and isinstance(model.base_model(), model_type))
    return contained or isinstance(model, model_type)


class EmbeddingTarget(Enum):
    Subject = 1,
    Predicate = 2,
    Object = 3


class EmbeddingType(Enum):
    Entity = 1,
    Relation = 2