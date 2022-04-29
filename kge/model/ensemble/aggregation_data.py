import torch
from torch.utils.data import Dataset, DataLoader, random_split

from kge.model.ensemble.model_manager import ModelManager
from kge.model.ensemble.util import EmbeddingType, EmbeddingTarget


def create_aggregation_dataloader(
    model_manager: ModelManager, e_type: EmbeddingType, split_ratio, batch_size, shuffle
):
    """
    Create training and validation dataloader for aggregation optimization.
    The dataset is created by the model manager for a given embedding type.
    Thereby, the split ration defines the split for the training and validation set.
    :param model_manager: Model Manager to create the set of embeddings.
    :param e_type: Target embedding type for aggregation optimizaiton.
    :param split_ratio: Split for the training dataset, such that 1-split_ration is size of validation dataset.
    :param batch_size: Batch size for both dataloader
    :param shuffle: Whether the data shall be shuffled during optimization.
    :return: Tuple of training and validation dataloader.
    """

    # Fetch embedding type specific data from the model manager
    if e_type == EmbeddingType.Entity:
        data = model_manager.fetch_model_embeddings(EmbeddingTarget.Subject)
    elif e_type == EmbeddingType.Relation:
        data = model_manager.fetch_model_embeddings(EmbeddingTarget.Predicate)
    else:
        raise ValueError("Unknown embedding type: " + str(e_type))
    dataset_complete = AggregationDataset(data)

    # split data into train and valid set
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
        Creates a new dataset for unsupervised learning of aggregation models, given a set of embeddings to learn.
        :param embeds: Embeddings for entities or relations to be used for aggregation training.
        """
        self.data = embeds

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        :return: Number of dataset samples.
        """
        return self.data[0].size()[0]

    def __getitem__(self, idx):
        """
        Return the sample at the specified position in the dataset.
        :param idx: Position of a sample.
        :return: Tuple of position and embedding sample.
        """
        item = {}
        for key, value in self.data.items():
            item[key] = value[idx]
        return idx, item
