import math
from collections import OrderedDict

import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from kge import Configurable, Config


class DimReductionBase(nn.Module, Configurable):

    def __init__(self, config: Config, configuration_key, parent_configuration_key):
        """
        Initializes basic dim reduction variables.
        Updates the reduced dimensionality in embedding ensemble if entity_reduction and relation_reduction
        are given for the reduction model, else do a concatenation
        :param config:
        :param configuration_key:
        :param parent_configuration_key:
        """
        Configurable.__init__(self, config, configuration_key)
        nn.Module.__init__(self)

        # update reduced embedding size in config in case of training
        if config.get("job.type") == "train":
            parent_configuration_key = parent_configuration_key
            num_models = len(config.get(parent_configuration_key + ".submodels"))
            entity_dim = config.get(parent_configuration_key + ".entities.source_dim")
            relation_dim = config.get(parent_configuration_key + ".relations.source_dim")
            entity_reduction = 1.0
            if self.has_option("entity_reduction"):
                entity_reduction = self.get_option("entity_reduction")
            relation_reduction = 1.0
            if self.has_option("relation_reduction"):
                relation_reduction = self.get_option("relation_reduction")
            entity_dim_reduced = math.floor(num_models * entity_dim * entity_reduction)
            relation_dim_reduced = math.floor(num_models * relation_dim * relation_reduction)
            self.config.set(parent_configuration_key + ".entities.reduced_dim", entity_dim_reduced)
            self.config.set(parent_configuration_key + ".relations.reduced_dim", relation_dim_reduced)

    def reduce_entities(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of entities, m is the number of submodels and dim_m is the
        length of the model embedding dimension
        :param t:
        :return:
        """
        raise NotImplementedError

    def reduce_relations(self, t: Tensor):
        """
        Execute a dimensionality reduction on the tensor of the form n times m times dim_m,
        where n is the number of relations, m is the number of submodels and dim_m is the
        length of the model embedding dimension
        :param t:
        :return:
        """
        raise NotImplementedError

    def train_dim_reduction(self, models):
        raise NotImplementedError


class ConcatenationReduction(DimReductionBase):

    def __init__(self, config, parent_configuration_key):
        DimReductionBase.__init__(self, config, None, parent_configuration_key)

    def reduce_entities(self, t: Tensor):
        n = t.size()[0]
        res = t.view(n, -1)
        return res

    def reduce_relations(self, t: Tensor):
        n = t.size()[0]
        res = t.view(n, -1)
        return res

    def train_dim_reduction(self, models):
        pass


class PcaReduction(DimReductionBase):

    def __init__(self, config, parent_configuration_key):
        DimReductionBase.__init__(self, config, "pca", parent_configuration_key)
        entity_source = config.get(parent_configuration_key + ".entities.source_dim")
        relation_source = config.get(parent_configuration_key + ".relations.source_dim")
        entity_dim = entity_source * self.get_option("entity_reduction")
        relation_dim = relation_source * self.get_option("relation_reduction")
        if entity_dim - int(entity_dim) > 0:
            raise Exception("PCA reduction of entities not doable for given percentage.")
        if relation_dim - int(relation_dim) > 0:
            raise Exception("PCA reduction of relations not doable for given percentage.")
        self.entity_pca = PCA(n_components=int(entity_dim))
        self.relation_pca = PCA(n_components=int(relation_dim))

    def reduce_entities(self, t: Tensor):
        t = torch.randn(2,128)
        self.entity_pca.fit(t)
        res = self.entity_pca.transform(t)
        return res

    def reduce_relations(self, t: Tensor):
        self.relation_pca.fit(t)
        res = self.relation_pca.transform(t)
        return res

    def train_dim_reduction(self, models):
        pass


class DimReductionDataset(Dataset):
    def __init__(self, models, mode="entity"):
        """
        Creates a new dataset for unsupervised learning of dimensionality reduction models.
        The data has the format n times m times dim_m
        :param models:
        :param mode: either "entity" or "relation"
        """
        self.data = None
        for idx, model in enumerate(models):
            if mode == "entity":
                # assuming subject and object embedder are the same
                m_embeds = model.get_s_embedder().embed_all().detach()
            elif mode == "relation":
                m_embeds = model.get_p_embedder().embed_all().detach()
            else:
                raise ValueError
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


class AutoencoderReduction(DimReductionBase):

    def __init__(self, config: Config, parent_configuration_key):
        DimReductionBase.__init__(self, config, "autoencoder_reduction", parent_configuration_key)

        if config.get("job.type") == "train":
            self.epochs = self.get_option("epochs")
            self.lr = self.get_option("lr")
            self.weight_decay = self.get_option("weight_decay")

        self.entity_model = Autoencoder(config, parent_configuration_key, "entities")
        self.relation_model = Autoencoder(config, parent_configuration_key, "relations")

    def reduce_entities(self, t: Tensor):
        n = t.size()[0]
        entities = t.view(n, -1)  # transform tensor to format
        entities = self.entity_model.reduce(entities)
        return entities

    def reduce_relations(self, t: Tensor):
        n = t.size()[0]
        relations = t.view(n, -1)  # transform tensor to format
        relations = self.relation_model.reduce(relations)
        return relations

    def train_dim_reduction(self, models):
        # create dataloader
        entity_dataloader = DataLoader(DimReductionDataset(models, "entity"), batch_size=10, shuffle=True)
        relation_dataloader = DataLoader(DimReductionDataset(models, "relation"), batch_size=10, shuffle=True)

        print("Training entity autoencoder")
        self.train_model(entity_dataloader, self.entity_model)

        print("Training relation autoencoder")
        self.train_model(relation_dataloader, self.relation_model)

    def train_model(self, dataloader, model):
        model.train()
        # validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        # using an Adam Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            loss_val = 0
            for batch in dataloader:
                n = batch.size()[0]
                embeds = batch.view(n, -1)

                # Output of Autoencoder
                reconstructed = model(embeds)

                # Calculating the loss function
                loss = loss_function(reconstructed, embeds)

                # The gradients are set to zero,
                # then the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss_val))


class Autoencoder(nn.Module, Configurable):
    def __init__(self, config: Config, parent_configuration_key, embedding_configuration_key):
        super(Autoencoder, self).__init__()
        Configurable.__init__(self, config, "autoencoder")
        num_models = len(config.get(parent_configuration_key + ".submodels"))
        source_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".source_dim")
        reduced_dim = config.get(parent_configuration_key + "." + embedding_configuration_key + ".reduced_dim")
        self.dim_in = num_models * source_dim
        self.dim_out = reduced_dim
        self.num_layers = self.get_option("num_layers")
        self.dropout = self.get_option("dropout")

        layer_dims = [
            round(self.dim_in - n * ((self.dim_in - self.dim_out) / self.num_layers)) for n in range(0, self.num_layers)
        ]
        layer_dims.append(self.dim_out)

        encode_dict = OrderedDict()
        for idx in range(0, self.num_layers):
            encode_dict["dropout" + str(idx)] = nn.Dropout(p=self.dropout)
            encode_dict["linear" + str(idx)] = nn.Linear(layer_dims[idx], layer_dims[idx + 1])
            if idx + 1 < self.num_layers:
                encode_dict["relu" + str(idx)] = nn.ReLU()

        decode_dict = OrderedDict()
        for idx in reversed(range(0, self.num_layers)):
            decode_dict["linear" + str(idx)] = nn.Linear(layer_dims[idx + 1], layer_dims[idx])
            if idx > 0:
                decode_dict["relu" + str(idx)] = nn.ReLU()

        self.encoder = torch.nn.Sequential(encode_dict)
        self.decoder = torch.nn.Sequential(decode_dict)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reduce(self, x):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded
