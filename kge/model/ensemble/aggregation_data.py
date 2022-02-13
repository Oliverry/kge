from torch.utils.data import Dataset, DataLoader

from kge.model.ensemble.model_manager import ModelManager


def create_aggregation_dataloader(model_manager: ModelManager, target, batch_size, shuffle):
    dataset = AggregationDataset(model_manager, target)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader


class AggregationDataset(Dataset):
    def __init__(self, model_manager: ModelManager, target):
        """
        Creates a new dataset for unsupervised learning of aggregation models.
        The data has the format n times m times dim_m
        :param models:
        :param target: either "entity" or "relation"
        """
        if target == "entities":
            self.data = model_manager.fetch_model_embeddings("s")
        elif target == "relations":
            self.data = model_manager.fetch_model_embeddings("p")
        else:
            raise ValueError("Unknown target embedding.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]
