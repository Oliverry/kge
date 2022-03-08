from torch.utils.data import Dataset, DataLoader

from kge.model.ensemble.model_manager import ModelManager
from kge.model.ensemble.util import EmbeddingType, EmbeddingTarget


def create_aggregation_dataloader(model_manager: ModelManager, e_type: EmbeddingType, batch_size, shuffle):
    dataset = AggregationDataset(model_manager, e_type)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader


class AggregationDataset(Dataset):
    def __init__(self, model_manager: ModelManager, e_type: EmbeddingType):
        """
        Creates a new dataset for unsupervised learning of aggregation models.
        The data has the format n times m times dim_m
        :param models:
        :param target:
        """
        if e_type == EmbeddingType.Entity:
            self.data = model_manager.fetch_model_embeddings(EmbeddingTarget.Subject)
        elif e_type == EmbeddingType.Relation:
            self.data = model_manager.fetch_model_embeddings(EmbeddingTarget.Predicate)
        else:
            raise ValueError("Unknown embedding type: " + str(e_type))

    def __len__(self):
        return self.data[0].size()[0]

    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            item[key] = value[idx]
        return idx, item
