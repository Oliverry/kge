import torch
from torch.utils.data import Dataset, DataLoader, random_split

from kge.model.ensemble.model_manager import ModelManager
from kge.model.ensemble.util import EmbeddingType, EmbeddingTarget


def create_aggregation_dataloader(
    model_manager: ModelManager, e_type: EmbeddingType, split_ratio, batch_size, shuffle
):
    if e_type == EmbeddingType.Entity:
        data = model_manager.fetch_model_embeddings(EmbeddingTarget.Subject)
    elif e_type == EmbeddingType.Relation:
        data = model_manager.fetch_model_embeddings(EmbeddingTarget.Predicate)
    else:
        raise ValueError("Unknown embedding type: " + str(e_type))
    dataset_complete = AggregationDataset(data)

    # split data to train and valid set
    n = len(dataset_complete)
    split_train = round(n * split_ratio)
    dataset_train, dataset_valid = random_split(
        dataset_complete, [split_train, n - split_train]
    )

    # create datasets and dataloader
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle)
    dataloader_valid = DataLoader(dataset_valid, batch_size, shuffle)

    return dataloader_train, dataloader_valid


class AggregationDataset(Dataset):
    def __init__(self, embeds):
        """
        Creates a new dataset for unsupervised learning of aggregation models.
        The data has the format n times m times dim_m
        :param models:
        :param target:
        """
        self.data = embeds

    def __len__(self):
        return self.data[0].size()[0]

    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            item[key] = value[idx]
        return idx, item
